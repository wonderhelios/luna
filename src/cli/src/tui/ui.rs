use ratatui::{
    prelude::*,
    widgets::{Block, Borders, Paragraph, Wrap},
};
use unicode_width::UnicodeWidthStr;

use super::state::{AppState, ChatRole};

pub fn draw(frame: &mut Frame, app: &AppState) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(3),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    let header = match (&app.session_id, app.busy) {
        (Some(id), true) => format!("session: {id} | Running..."),
        (Some(id), false) => format!("session: {id}"),
        (None, true) => "session: <new> | Running...".to_owned(),
        (None, false) => "session: <new>".to_owned(),
    };

    let chat_text = build_chat_text(app);
    let chat = Paragraph::new(chat_text)
        .block(Block::default().borders(Borders::ALL).title(header))
        .wrap(Wrap { trim: false })
        .scroll((app.scroll_y as u16, 0));
    frame.render_widget(chat, chunks[0]);

    // Input line
    let (input_view, cursor_x) =
        input_view_and_cursor(&app.input, app.input_cursor, chunks[1].width);
    let input =
        Paragraph::new(input_view).block(Block::default().borders(Borders::ALL).title("输入"));
    frame.render_widget(input, chunks[1]);
    // Place cursor inside the input box.
    let x = chunks[1].x + 1 + cursor_x;
    let y = chunks[1].y + 1;
    frame.set_cursor_position((x, y));

    // Status bar
    let status = if app.status.is_empty() {
        "Ctrl+C 退出 | Enter 发送 | PgUp/PgDn 滚动".to_owned()
    } else {
        app.status.clone()
    };
    let status = Paragraph::new(status).block(Block::default());
    frame.render_widget(status, chunks[2]);
}

fn build_chat_text(app: &AppState) -> Text<'_> {
    let mut lines: Vec<Line> = Vec::new();
    for m in &app.messages {
        let prefix = match m.role {
            ChatRole::User => "You: ",
            ChatRole::Assistant => "Luna: ",
            ChatRole::System => "",
        };
        let mut first = true;
        for l in m.content.lines() {
            if first {
                lines.push(Line::from(vec![Span::raw(format!("{prefix}{l}"))]));
                first = false;
            } else {
                lines.push(Line::from(Span::raw(l.to_owned())));
            }
        }
        lines.push(Line::from(Span::raw(""))); // blank line between messages
    }
    Text::from(lines)
}

fn input_view_and_cursor(input: &str, cursor: usize, width: u16) -> (String, u16) {
    // width includes borders; inner width is at least 1.
    let inner = width.saturating_sub(2).max(1) as usize;
    let chars: Vec<char> = input.chars().collect();
    let cursor = cursor.min(chars.len());
    let full = chars.iter().collect::<String>();

    // If it fits, show all.
    if UnicodeWidthStr::width(full.as_str()) <= inner {
        let x = UnicodeWidthStr::width(chars[..cursor].iter().collect::<String>().as_str()) as u16;
        return (full, x);
    }

    // Otherwise, show a tail window that includes the cursor.
    let mut start = 0usize;
    while start < cursor {
        let view: String = chars[start..].iter().collect();
        if UnicodeWidthStr::width(view.as_str()) <= inner {
            break;
        }
        start += 1;
    }
    let mut end = chars.len();
    while end > start {
        let view: String = chars[start..end].iter().collect();
        if UnicodeWidthStr::width(view.as_str()) <= inner {
            break;
        }
        end -= 1;
    }
    let view: String = chars[start..end].iter().collect();
    let x = UnicodeWidthStr::width(chars[start..cursor].iter().collect::<String>().as_str()) as u16;
    (view, x)
}
