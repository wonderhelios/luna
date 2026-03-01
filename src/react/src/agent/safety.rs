//! ReAct agent safety state and helper methods
//!
//! Used to centrally store "safety"-related state in the ReAct loop, such as:
//! - Consecutive no-delta search count (used to determine if should stop continuing search)
//! - Most recent search query (used to detect duplicate searches)
//! - Most recent edit location (file + line range, used to detect duplicate edits)

use serde::{Deserialize, Serialize};

/// Safety-related state in the ReAct loop
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ReActSafetyState {
    /// Consecutive no-delta search count
    pub no_delta_searches: usize,
    /// Most recent search query (used to detect duplicate searches)
    pub last_search_query: Option<String>,
    /// Most recent edit location (path, start_line, end_line)
    pub last_edit: Option<(String, usize, usize)>,
}

impl ReActSafetyState {
    /// Whether to "answer directly" based on current safety state (e.g., multiple searches with no delta and already has context)
    pub fn should_auto_answer(&self, has_context: bool) -> bool {
        self.no_delta_searches >= 2 && has_context
    }

    /// Record a search and return: whether the search is duplicate, whether it has no delta
    pub fn record_search(&mut self, query: &str, had_delta: bool) -> (bool, bool) {
        let repeated = self
            .last_search_query
            .as_deref()
            .map(|last| last.eq_ignore_ascii_case(query))
            .unwrap_or(false);

        if had_delta {
            self.no_delta_searches = 0;
        } else {
            self.no_delta_searches += 1;
        }

        self.last_search_query = Some(query.to_string());
        let no_delta = !had_delta;
        (repeated, no_delta)
    }

    /// Detect if this edit is exactly the same as the previous edit (same file, same range)
    pub fn is_duplicate_edit(&self, path: &str, start_line: usize, end_line: usize) -> bool {
        if let Some((last_path, last_start, last_end)) = &self.last_edit {
            last_path == path && *last_start == start_line && *last_end == end_line
        } else {
            false
        }
    }

    /// Record an edit location
    pub fn record_edit(&mut self, path: String, start_line: usize, end_line: usize) {
        self.last_edit = Some((path, start_line, end_line));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_search_updates_counters_and_flags() {
        let mut s = ReActSafetyState::default();

        // First search, with delta: not counted in no_delta, repeated=false
        let (repeated, no_delta) = s.record_search("foo", true);
        assert!(!repeated);
        assert!(!no_delta);
        assert_eq!(s.no_delta_searches, 0);

        // Second same query, no delta: repeated=true, no_delta=true, counted in no_delta_searches
        let (repeated2, no_delta2) = s.record_search("foo", false);
        assert!(repeated2);
        assert!(no_delta2);
        assert_eq!(s.no_delta_searches, 1);
    }

    #[test]
    fn should_auto_answer_after_multiple_no_delta_searches() {
        let mut s = ReActSafetyState::default();

        // Two consecutive no-delta searches
        s.record_search("foo", false);
        s.record_search("foo", false);

        // Auto answer only triggers when there is context
        assert!(s.should_auto_answer(true));
        assert!(!s.should_auto_answer(false));
    }

    #[test]
    fn duplicate_edit_detection_works() {
        let mut s = ReActSafetyState::default();

        assert!(!s.is_duplicate_edit("path.rs", 10, 20));
        s.record_edit("path.rs".to_string(), 10, 20);

        assert!(s.is_duplicate_edit("path.rs", 10, 20));
        assert!(!s.is_duplicate_edit("path.rs", 11, 20));
        assert!(!s.is_duplicate_edit("other.rs", 10, 20));
    }
}
