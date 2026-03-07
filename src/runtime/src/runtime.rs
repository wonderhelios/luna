use error::{LunaError, Result, ResultExt as _};
use session::Role;

use std::path::Path;

use crate::{
    command,
    config::RuntimeConfig,
    recorder::{TrajectoryEvent, TrajectoryStep},
    request::{RunRequest, SessionRef},
    response::{RunResponse, RuntimeEvent},
};

pub struct LunaRuntime {
    config: RuntimeConfig,
}

impl LunaRuntime {
    pub fn new() -> Self {
        Self {
            config: RuntimeConfig::default(),
        }
    }

    pub fn with_config(config: RuntimeConfig) -> Self {
        Self { config }
    }

    pub async fn run(&self, req: RunRequest) -> Result<RunResponse> {
        let trajectory = self.config.trajectory();
        trajectory.on_run_start(&req);

        let mut session_id_for_end: Option<String> = None;
        let result = self.run_impl(req, &mut session_id_for_end);

        trajectory.on_run_end(session_id_for_end.as_deref(), result.as_ref());

        result
    }

    pub fn run_impl(
        &self,
        req: RunRequest,
        session_id_for_end: &mut Option<String>,
    ) -> Result<RunResponse> {
        let RunRequest {
            request_id,
            session,
            input: user_input,
            cwd,
            ..
        } = req;

        let session_store = self.config.session_store();
        let trajectory = self.config.trajectory();

        let mut events = Vec::<RuntimeEvent>::new();

        // Phase-2: slash commands.
        if let Some(cmd) = command::parse_slash_command(&user_input) {
            // Ensure we always return a usable session id to CLI.
            // If user hasn't started a session yet, we create one but do not append messages.
            let current_session_id = match &session {
                SessionRef::Existing { session_id } => session_id.clone(),
                SessionRef::New { title } => {
                    let s = session_store
                        .create(title.clone())
                        .context("create session")?;
                    trajectory.on_event(&s.id, TrajectoryEvent::SessionCreated);
                    events.push(RuntimeEvent::SessionCreated {
                        session_id: s.id.clone(),
                    });
                    s.id
                }
            };
            *session_id_for_end = Some(current_session_id.clone());

            match cmd {
                command::Command::Sessions => {
                    let out = command::render_sessions_list(
                        Some(&current_session_id),
                        session_store.as_ref(),
                    )?;
                    trajectory.on_step(&TrajectoryStep {
                        ts_ms: now_ms(),
                        session_id: current_session_id.clone(),
                        request_id: request_id.clone(),
                        state: serde_json::json!({ "cwd": cwd, "type": "command" }),
                        action: serde_json::json!({ "type": "command", "name": "/sessions" }),
                        reward: 0.2,
                        outcome: serde_json::json!({ "ok": true, "output_len": out.len() }),
                    });
                    return Ok(RunResponse {
                        request_id,
                        session_id: current_session_id,
                        output: out,
                        events,
                    });
                }
                command::Command::Switch { session_id } => {
                    // Validate target session exists.
                    // UX: allow users to paste either full id (e.g. `local:<uuid>`) or the uuid only.
                    let candidates: Vec<String> = if session_id.contains(':') {
                        vec![session_id]
                    } else {
                        vec![format!("local:{session_id}"), session_id]
                    };

                    let mut chosen_target: Option<String> = None;
                    for cand in &candidates {
                        if session_store.get(cand).context("load session")?.is_some() {
                            chosen_target = Some(cand.clone());
                            break;
                        }
                    }
                    let ok = chosen_target.is_some();
                    let target = chosen_target.unwrap_or_else(|| candidates[0].clone());
                    let out = if ok {
                        format!("Switched to session: {target}\n")
                    } else {
                        format!("❌ session not found: {target}\n")
                    };
                    let out_len = out.len();

                    let chosen = if ok { target } else { current_session_id };
                    *session_id_for_end = Some(chosen.clone());

                    trajectory.on_step(&TrajectoryStep {
                        ts_ms: now_ms(),
                        session_id: chosen.clone(),
                        request_id: request_id.clone(),
                        state: serde_json::json!({ "cwd": cwd, "type": "command" }),
                        action: serde_json::json!({ "type": "command", "name": "/switch" }),
                        reward: if ok { 0.2 } else { -0.5 },
                        outcome: serde_json::json!({ "ok": ok, "output_len": out_len }),
                    });
                    return Ok(RunResponse {
                        request_id,
                        session_id: chosen,
                        output: out,
                        events,
                    });
                }
            }
        }

        // 1) load or create session
        let mut session = match session {
            SessionRef::New { title } => {
                let session = session_store.create(title).context("create session")?;
                trajectory.on_event(&session.id, TrajectoryEvent::SessionCreated);
                events.push(RuntimeEvent::SessionCreated {
                    session_id: session.id.clone(),
                });
                session
            }
            SessionRef::Existing { session_id } => {
                let session = session_store
                    .get(&session_id)
                    .context("load session")?
                    .ok_or_else(|| {
                        LunaError::not_found(format!("session not found: {session_id}"))
                    })?;
                trajectory.on_event(&session.id, TrajectoryEvent::SessionLoaded);
                events.push(RuntimeEvent::SessionLoaded {
                    session_id: session.id.clone(),
                });
                session
            }
        };
        let session_id = session.id.clone();
        *session_id_for_end = Some(session_id.clone());

        // 2) append user message
        session.push_message(Role::User, &user_input);
        trajectory.on_event(
            &session.id,
            TrajectoryEvent::MessageAppended {
                role: Role::User,
                bytes: user_input.len(),
            },
        );
        events.push(RuntimeEvent::UserMessageAppended);

        // 3) produce assistant output
        let output = self.produce_output(
            &session_id,
            &request_id,
            &user_input,
            cwd.as_deref(),
            &mut events,
        )?;

        // 4) append assistant message
        session.push_message(Role::Assistant, &output);
        trajectory.on_event(
            &session.id,
            TrajectoryEvent::MessageAppended {
                role: Role::Assistant,
                bytes: output.len(),
            },
        );
        events.push(RuntimeEvent::AssistantMessageAppended);

        // 5) save session
        session_store.save(session).context("save session")?;

        // Phase-2: record a minimal trajectory step for this turn.
        let reward = match events
            .iter()
            .filter_map(|e| match e {
                RuntimeEvent::ScopeGraphSearchCompleted { matches } => Some(*matches),
                _ => None,
            })
            .max()
        {
            Some(m) if m > 0 => 1.0,
            Some(_) => 0.0,
            None => 0.0,
        };
        trajectory.on_step(&TrajectoryStep {
            ts_ms: now_ms(),
            session_id: session_id.clone(),
            request_id: request_id.clone(),
            state: serde_json::json!({ "cwd": cwd, "input_len": user_input.len() }),
            action: serde_json::json!({ "type": "chat_turn" }),
            reward,
            outcome: serde_json::json!({ "output_len": output.len() }),
        });

        Ok(RunResponse {
            request_id,
            session_id,
            output,
            events,
        })
    }

    fn produce_output(
        &self,
        session_id: &str,
        request_id: &str,
        user_input: &str,
        cwd: Option<&Path>,
        events: &mut Vec<RuntimeEvent>,
    ) -> Result<String> {
        crate::tpar::run_turn(
            user_input,
            crate::tpar::TurnContext {
                session_id: session_id.to_owned(),
                request_id: request_id.to_owned(),
                cwd: cwd.map(|p| p.to_path_buf()),
                safety_guard: self.config.safety(),
                trajectory: self.config.trajectory(),
                tools: self.config.tools(),
                budget: self.config.budget(),
            },
            events,
        )
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as u64
}

impl Default for LunaRuntime {
    fn default() -> Self {
        Self::new()
    }
}
