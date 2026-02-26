//! ReAct agent 安全状态与辅助方法
//!
//! 用于在 ReAct 循环中集中保存与“安全”相关的状态，例如：
//! - 连续无增量搜索次数（用于判断是否该停止继续 search）
//! - 最近一次搜索的 query（用于识别重复搜索）
//! - 最近一次编辑的位置（文件 + 行范围，用于识别重复编辑）

use serde::{Deserialize, Serialize};

/// ReAct 循环中的安全相关状态
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ReActSafetyState {
    /// 连续无增量搜索次数
    pub no_delta_searches: usize,
    /// 最近一次搜索的 query（用于检测重复搜索）
    pub last_search_query: Option<String>,
    /// 最近一次编辑的位置（path, start_line, end_line）
    pub last_edit: Option<(String, usize, usize)>,
}

impl ReActSafetyState {
    /// 是否应该基于当前安全状态“直接回答”（例如多次 search 无增量且已经有上下文）
    pub fn should_auto_answer(&self, has_context: bool) -> bool {
        self.no_delta_searches >= 2 && has_context
    }

    /// 记录一次搜索，并返回：该搜索是否重复、是否无增量
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

    /// 检测本次编辑是否与上一次编辑完全相同（同文件同范围）
    pub fn is_duplicate_edit(&self, path: &str, start_line: usize, end_line: usize) -> bool {
        if let Some((last_path, last_start, last_end)) = &self.last_edit {
            last_path == path && *last_start == start_line && *last_end == end_line
        } else {
            false
        }
    }

    /// 记录一次编辑位置
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

        // 第一次搜索，有增量：不计入 no_delta，repeated=false
        let (repeated, no_delta) = s.record_search("foo", true);
        assert!(!repeated);
        assert!(!no_delta);
        assert_eq!(s.no_delta_searches, 0);

        // 第二次同 query，无增量：repeated=true，no_delta=true，计入 no_delta_searches
        let (repeated2, no_delta2) = s.record_search("foo", false);
        assert!(repeated2);
        assert!(no_delta2);
        assert_eq!(s.no_delta_searches, 1);
    }

    #[test]
    fn should_auto_answer_after_multiple_no_delta_searches() {
        let mut s = ReActSafetyState::default();

        // 连续两次无增量搜索
        s.record_search("foo", false);
        s.record_search("foo", false);

        // 有上下文时才会触发 auto answer
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
