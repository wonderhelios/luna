use anyhow::Context;
use session::Role;

use std::path::Path;

use crate::{
    config::RuntimeConfig,
    recorder::TrajectoryEvent,
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

    pub async fn run(&self, req: RunRequest) -> anyhow::Result<RunResponse> {
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
    ) -> anyhow::Result<RunResponse> {
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
                    .ok_or_else(|| anyhow::anyhow!("session not found: {session_id}"))?;
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
        let output = self.produce_output(&user_input, cwd.as_deref(), &mut events)?;

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

        Ok(RunResponse {
            request_id,
            session_id,
            output,
            events,
        })
    }

    fn produce_output(
        &self,
        user_input: &str,
        cwd: Option<&Path>,
        events: &mut Vec<RuntimeEvent>,
    ) -> anyhow::Result<String> {
        // Phase-1: intelligence-first for symbol navigation queries.
        let router = crate::router::RuntimeRouter::default();
        if let Some(out) = router.maybe_handle(user_input, cwd, events)? {
            return Ok(out);
        }

        Ok(format!("received: {user_input}"))
    }
}

impl Default for LunaRuntime {
    fn default() -> Self {
        Self::new()
    }
}
