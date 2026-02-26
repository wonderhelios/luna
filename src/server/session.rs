use std::collections::HashMap;

use toolkit::ExecutionPolicy;
use uuid::Uuid;

use session::FileSessionStore;
use session::SessionStore as SessionBackend;
pub use session::{PendingToolCall, SessionMetadata, SessionState};

use crate::rpc::{rpc_err, RpcErrorCode};
use crate::util::session_id_from_params;

#[derive(Debug)]
pub struct SessionStore {
    sessions: HashMap<String, SessionState>,
    current_session_id: String,
    backend: FileSessionStore,
}

impl SessionStore {
    pub fn new() -> Self {
        let base_dir =
            std::env::var("LUNA_SESSION_DIR").unwrap_or_else(|_| ".luna/sessions".to_string());
        let backend = FileSessionStore::new(&base_dir)
            .unwrap_or_else(|e| panic!("failed to create FileSessionStore at {}: {}", base_dir, e));

        let mut sessions = HashMap::new();
        let current_session_id = Uuid::new_v4().to_string();
        let initial_state = SessionState {
            policy: ExecutionPolicy::default(),
            pending: HashMap::new(),
            metadata: SessionMetadata::default(),
        };
        sessions.insert(current_session_id.clone(), initial_state.clone());
        let _ = backend.insert(current_session_id.clone(), initial_state);

        Self {
            sessions,
            current_session_id,
            backend,
        }
    }

    pub fn resolve_or_create(
        &mut self,
        method: &str,
        params: &serde_json::Value,
    ) -> anyhow::Result<String> {
        let provided_sid = session_id_from_params(params);
        let sid = provided_sid
            .clone()
            .unwrap_or_else(|| self.current_session_id.clone());

        if !self.sessions.contains_key(&sid) {
            if method == "initialize" {
                let state = SessionState {
                    policy: ExecutionPolicy::default(),
                    pending: HashMap::new(),
                    metadata: SessionMetadata::default(),
                };
                self.sessions.insert(sid.clone(), state.clone());
                let _ = self.backend.insert(sid.clone(), state);
            } else {
                return Err(rpc_err(
                    RpcErrorCode::UnknownSession,
                    format!("unknown session_id: {sid}"),
                ));
            }
        }

        self.current_session_id = sid.clone();
        Ok(sid)
    }

    pub fn get(&self, sid: &str) -> Option<&SessionState> {
        self.sessions.get(sid)
    }

    pub fn get_mut(&mut self, sid: &str) -> Option<&mut SessionState> {
        self.sessions.get_mut(sid)
    }

    pub fn upsert(&mut self, sid: String, state: SessionState) {
        self.sessions.insert(sid.clone(), state.clone());
        let _ = if self.backend.contains(&sid).unwrap_or(false) {
            self.backend.update(&sid, state)
        } else {
            self.backend.insert(sid, state)
        };
    }
}
