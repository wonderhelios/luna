use std::{
    fs,
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::{now_ms, Message, Result, Session, SessionStore, SessionSummary, TimestampMs};

#[derive(Debug, Clone)]
pub struct LunaHome {
    base_dir: PathBuf,
}

impl LunaHome {
    pub fn from_env() -> Option<Self> {
        if let Some(v) = std::env::var_os("LUNA_HOME") {
            let p = PathBuf::from(v);
            return Some(Self { base_dir: p });
        }

        let home = std::env::var_os("HOME")
            .or_else(|| std::env::var_os("USERPROFILE"))
            .map(PathBuf::from)?;
        Some(Self {
            base_dir: home.join(".luna"),
        })
    }

    pub fn base_dir(&self) -> &Path {
        &self.base_dir
    }

    pub fn sessions_dir(&self) -> PathBuf {
        self.base_dir.join("sessions")
    }

    pub fn trajectories_dir(&self) -> PathBuf {
        self.base_dir.join("trajectories")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum SessionEvent {
    SessionCreated {
        session_id: String,
        title: Option<String>,
        ts_ms: TimestampMs,
    },
    MessageAppended {
        message: Message,
    },
}

/// A simple append-only jsonl session store
///
/// File layout: `~/.luna/sessions/<session_id>.jsonl`
#[derive(Debug)]
pub struct JsonlSessionStore {
    home: LunaHome,
    // Cache of persisted message counts per session id
    persisted_counts: Mutex<std::collections::HashMap<String, usize>>,
}

impl JsonlSessionStore {
    pub fn new(home: LunaHome) -> Self {
        Self {
            home,
            persisted_counts: Mutex::new(std::collections::HashMap::new()),
        }
    }

    pub fn try_default() -> Option<Self> {
        let home = LunaHome::from_env()?;
        Some(Self::new(home))
    }

    fn sessions_dir(&self) -> PathBuf {
        self.home.sessions_dir()
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir().join(format!("{session_id}.jsonl"))
    }

    fn ensure_dirs(&self) -> Result<()> {
        fs::create_dir_all(self.sessions_dir())?;
        Ok(())
    }

    fn append_event(&self, session_id: &str, event: &SessionEvent) -> Result<()> {
        self.ensure_dirs()?;
        let path = self.session_path(session_id);
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        let line = serde_json::to_string(event)?;
        f.write_all(line.as_bytes())?;
        f.write_all(b"\n")?;
        Ok(())
    }

    fn replay_session_file(&self, session_id: &str) -> Result<Option<Session>> {
        let path = self.session_path(session_id);
        let f = match fs::File::open(&path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        let mut session: Option<Session> = None;
        let mut message_count: usize = 0;
        for line in BufReader::new(f).lines() {
            let line = match line {
                Ok(l) => l,
                Err(e) => {
                    tracing::warn!("skip unreadable sesion line: id={session_id}, err={e}");
                    continue;
                }
            };
            if line.trim().is_empty() {
                continue;
            }
            let ev: SessionEvent = match serde_json::from_str(&line) {
                Ok(v) => v,
                Err(e) => {
                    tracing::warn!(
                        "skip invalid session jsonl line: id={session_id}, err={e}, line={line:?}"
                    );
                    continue;
                }
            };
            match ev {
                SessionEvent::SessionCreated {
                    session_id: _,
                    title,
                    ts_ms,
                } => {
                    session = Some(Session {
                        id: session_id.to_owned(),
                        title,
                        messages: Vec::new(),
                        created_at: ts_ms,
                        update_at: ts_ms,
                    });
                }
                SessionEvent::MessageAppended { message } => {
                    if let Some(s) = session.as_mut() {
                        s.update_at = s.update_at.max(message.timestamp);
                        s.messages.push(message);
                        message_count += 1;
                    }
                }
            }
        }

        if let Some(s) = &session {
            self.persisted_counts
                .lock()
                .insert(s.id.clone(), message_count);
        }

        Ok(session)
    }
}
impl SessionStore for JsonlSessionStore {
    fn get(&self, id: &str) -> Result<Option<Session>> {
        self.replay_session_file(id)
    }

    fn create(&self, title: Option<String>) -> Result<Session> {
        let session_id = crate::gen_id("local");
        let now = now_ms();
        let session = Session {
            id: session_id.clone(),
            title: title.clone(),
            messages: Vec::new(),
            created_at: now,
            update_at: now,
        };
        self.append_event(
            &session_id,
            &SessionEvent::SessionCreated {
                session_id: session_id.clone(),
                title,
                ts_ms: now,
            },
        )?;
        self.persisted_counts.lock().insert(session_id, 0);
        Ok(session)
    }

    fn save(&self, session: Session) -> Result<()> {
        let mut guard = self.persisted_counts.lock();
        let persisted = match guard.get(&session.id).copied() {
            Some(v) => v,
            None => {
                // Populate cache by replaying once.
                drop(guard);
                let _ = self.replay_session_file(&session.id)?;
                guard = self.persisted_counts.lock();
                guard.get(&session.id).copied().unwrap_or(0)
            }
        };

        if persisted >= session.messages.len() {
            // Nothing new.
            return Ok(());
        }

        for m in session.messages.iter().skip(persisted) {
            self.append_event(
                &session.id,
                &SessionEvent::MessageAppended { message: m.clone() },
            )?;
        }
        guard.insert(session.id.clone(), session.messages.len());
        Ok(())
    }

    fn list(&self) -> Result<Vec<SessionSummary>> {
        self.ensure_dirs()?;
        let mut out = Vec::new();
        for entry in fs::read_dir(self.sessions_dir())? {
            let entry = match entry {
                Ok(e) => e,
                Err(err) => {
                    tracing::warn!("skip unreadable sessions dir entry: {err}");
                    continue;
                }
            };
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
                continue;
            }
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let id = stem.to_owned();
            let Some(s) = self.replay_session_file(&id)? else {
                continue;
            };
            out.push(SessionSummary {
                id: s.id,
                title: s.title,
                message_count: s.messages.len(),
                updated_at: s.update_at,
            });
        }
        out.sort_by(|a, b| b.updated_at.cmp(&a.updated_at));
        Ok(out)
    }
}
