//! Session management

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type Result<T> = anyhow::Result<T>;
pub type TimestampMs = u64;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Session {
    pub id: String,
    pub title: Option<String>,
    pub messages: Vec<Message>,
    pub created_at: TimestampMs,
    pub update_at: TimestampMs,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Message {
    pub id: String,
    pub role: Role,
    pub content: String,
    pub timestamp: TimestampMs,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub enum Role {
    User,
    Assistant,
    System,
}

impl Session {
    pub fn new(id: impl Into<String>, title: Option<String>) -> Self {
        let now = now_ms();
        Self {
            id: id.into(),
            title,
            messages: Vec::new(),
            created_at: now,
            update_at: now,
        }
    }

    pub fn push_message(&mut self, role: Role, content: impl Into<String>) -> &Message {
        let now = now_ms();
        let message = Message {
            id: gen_id("msg"),
            role,
            content: content.into(),
            timestamp: now,
        };
        self.messages.push(message);
        self.update_at = now;
        self.messages.last().expect("just pushed")
    }
}

pub trait SessionStore: Send + Sync {
    fn get(&self, id: &str) -> Result<Option<Session>>;
    fn create(&self, title: Option<String>) -> Result<Session>;
    fn save(&self, session: Session) -> Result<()>;
}

#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    inner: Mutex<HashMap<String, Session>>,
}

impl InMemorySessionStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl SessionStore for InMemorySessionStore {
    fn get(&self, id: &str) -> Result<Option<Session>> {
        let guard = self.inner.lock();

        Ok(guard.get(id).cloned())
    }

    fn create(&self, title: Option<String>) -> Result<Session> {
        let mut guard = self.inner.lock();
        let id = gen_id("local");
        let session = Session::new(id.clone(), title);
        guard.insert(id.clone(), session.clone());

        Ok(session)
    }

    fn save(&self, session: Session) -> Result<()> {
        let mut guard = self.inner.lock();
        guard.insert(session.id.clone(), session);

        Ok(())
    }
}

/// Generate a reasonably unique id
pub fn gen_id(prefix: &str) -> String {
    format!("{prefix}:{}", uuid::Uuid::new_v4())
}

fn now_ms() -> TimestampMs {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_micros() as TimestampMs
}
