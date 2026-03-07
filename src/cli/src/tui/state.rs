use std::path::PathBuf;
use std::sync::Arc;

use runtime::LunaRuntime;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
    System,
}

#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

pub struct AppState {
    pub runtime: Arc<LunaRuntime>,
    pub cwd: Option<PathBuf>,
    pub session_id: Option<String>,
    pub messages: Vec<ChatMessage>,

    pub input: String,
    pub input_cursor: usize, // char index

    pub scroll_y: usize,
    pub busy: bool,
    pub status: String,
}

impl AppState {
    pub fn new(runtime: Arc<LunaRuntime>, cwd: Option<PathBuf>) -> Self {
        Self {
            runtime,
            cwd,
            session_id: None,
            messages: vec![ChatMessage {
                role: ChatRole::System,
                content: "🌙 Luna - AI Code Assistant\nCtrl+C 退出 | Enter 发送 | PgUp/PgDn 滚动"
                    .to_owned(),
            }],
            input: String::new(),
            input_cursor: 0,
            scroll_y: 0,
            busy: false,
            status: String::new(),
        }
    }

    pub fn push_user(&mut self, text: String) {
        self.messages.push(ChatMessage {
            role: ChatRole::User,
            content: text,
        });
    }

    pub fn push_assistant(&mut self, text: String) {
        self.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: text,
        });
    }

    pub fn push_system(&mut self, text: String) {
        self.messages.push(ChatMessage {
            role: ChatRole::System,
            content: text,
        });
    }

    pub fn clear_input(&mut self) {
        self.input.clear();
        self.input_cursor = 0;
    }

    pub fn input_insert(&mut self, ch: char) {
        let mut chars: Vec<char> = self.input.chars().collect();
        let idx = self.input_cursor.min(chars.len());
        chars.insert(idx, ch);
        self.input_cursor = idx + 1;
        self.input = chars.into_iter().collect();
    }

    pub fn input_backspace(&mut self) {
        if self.input_cursor == 0 {
            return;
        }
        let mut chars: Vec<char> = self.input.chars().collect();
        if self.input_cursor <= chars.len() {
            chars.remove(self.input_cursor - 1);
            self.input_cursor -= 1;
            self.input = chars.into_iter().collect();
        }
    }

    pub fn input_move_left(&mut self) {
        self.input_cursor = self.input_cursor.saturating_sub(1);
    }

    pub fn input_move_right(&mut self) {
        let len = self.input.chars().count();
        self.input_cursor = (self.input_cursor + 1).min(len);
    }
}
