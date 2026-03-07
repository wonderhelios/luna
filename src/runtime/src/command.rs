use error::Result;
use session::SessionStore;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Command {
    Sessions,
    Switch { session_id: String },
}

pub fn parse_slash_command(input: &str) -> Option<Command> {
    let s = input.trim();
    if !s.starts_with('/') {
        return None;
    }
    let mut parts = s.split_whitespace();
    let cmd = parts.next().unwrap_or("");
    match cmd {
        "/sessions" => Some(Command::Sessions),
        "/switch" => {
            let id = parts.next()?.to_owned();
            Some(Command::Switch { session_id: id })
        }
        _ => None,
    }
}

pub fn render_sessions_list(current: Option<&str>, store: &dyn SessionStore) -> Result<String> {
    let sessions = store.list()?;
    let mut out = String::new();

    out.push_str("Active sessions:\n");
    if sessions.is_empty() {
        out.push_str(" (none)\n");
        return Ok(out);
    }
    for s in sessions {
        let title = s.title.as_deref().unwrap_or("(untitled)");
        let mark = current
            .filter(|cur| *cur == s.id)
            .map(|_| " [current]")
            .unwrap_or("");
        out.push_str(&format!(
            " {} - {} ({} messages){}\n",
            s.id, title, s.message_count, mark
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_sessions() {
        assert_eq!(parse_slash_command("/sessions"), Some(Command::Sessions));
        assert_eq!(parse_slash_command(" /sessions  "), Some(Command::Sessions));
    }

    #[test]
    fn parse_switch() {
        assert_eq!(
            parse_slash_command("/switch local:abc"),
            Some(Command::Switch {
                session_id: "local:abc".to_owned()
            })
        );
        assert_eq!(
            parse_slash_command("/switch abc"),
            Some(Command::Switch {
                session_id: "abc".to_owned()
            })
        );
        assert_eq!(parse_slash_command("/switch"), None);
    }
}
