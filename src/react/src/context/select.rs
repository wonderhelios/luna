//! Context Engine: 命中合并与选择

use anyhow::Result;
use core::code_chunk::ContextChunk;
use std::collections::BTreeMap;
use std::path::Path;
use tokenizers::Tokenizer;

use super::render::ContextEngineOptions;
use tools::fs::read_file_by_lines;

/// 根据 hits 选择、合并、排序 ContextChunk
pub fn select_context_chunks(
    repo_root: &Path,
    hits: &[core::code_chunk::IndexChunk],
    context: &[ContextChunk],
    tokenizer: &Tokenizer,
    opt: ContextEngineOptions,
) -> Result<Vec<ContextChunk>> {
    // 1) 先按 (path, start_line, end_line) 合并上下文
    let mut by_path: BTreeMap<String, Vec<ContextChunk>> = BTreeMap::new();
    for c in context {
        by_path.entry(c.path.clone()).or_default().push(c.clone());
    }

    let mut merged_all = Vec::new();
    for (path, mut chunks) in by_path {
        chunks.sort_by_key(|c| (c.start_line, c.end_line));
        let mut merged: Vec<(usize, usize, String)> = Vec::new();
        for c in chunks {
            let (s, e) = (c.start_line, c.end_line);
            if let Some((_ms, me, reason)) = merged.last_mut() {
                let gap_ok = s <= me.saturating_add(opt.merge_gap_lines + 1);
                if gap_ok {
                    *me = (*me).max(e);
                    if !c.reason.is_empty() && !reason.contains(&c.reason) {
                        if !reason.is_empty() {
                            reason.push_str("; ");
                        }
                        reason.push_str(&c.reason);
                    }
                    continue;
                }
            }
            merged.push((s, e, c.reason));
        }
        for (s, e, reason) in merged {
            let snippet = read_file_by_lines(repo_root, &path, s, e)?;
            merged_all.push(ContextChunk {
                path: path.clone(),
                alias: 0,
                snippet,
                start_line: s,
                end_line: e,
                reason,
            });
        }
    }

    // 2) 根据 hits 计算每个 ContextChunk 的 hit 数，作为排序依据
    let mut scored = merged_all
        .into_iter()
        .map(|c| {
            let mut cnt = 0usize;
            for h in hits {
                if h.path == c.path && h.start_line >= c.start_line && h.end_line <= c.end_line {
                    cnt += 1;
                }
            }
            (cnt, c)
        })
        .collect::<Vec<_>>();

    scored.sort_by(|(ac, a), (bc, b)| {
        bc.cmp(ac)
            .then_with(|| {
                let asz = a.end_line.saturating_sub(a.start_line);
                let bsz = b.end_line.saturating_sub(b.start_line);
                asz.cmp(&bsz)
            })
            .then_with(|| a.path.cmp(&b.path))
            .then_with(|| a.start_line.cmp(&b.start_line))
    });

    let mut selected = scored
        .into_iter()
        .map(|(_, c)| c)
        .take(opt.max_chunks.max(1))
        .collect::<Vec<_>>();

    // 3) 按 token budget 进行裁剪
    if opt.max_total_tokens > 0 {
        let mut total = 0usize;
        let mut keep = Vec::new();
        for c in &selected {
            let s = format!("{}\n{}", c.path, c.snippet);
            let t = tokenizer.encode(s, true).map(|e| e.len()).unwrap_or(0);
            if total + t <= opt.max_total_tokens {
                total += t;
                keep.push(c.clone());
            }
        }
        selected = keep;
    }

    // 4) 规范 alias，并按 path+start_line 稳定排序
    selected.sort_by(|a, b| (a.path.as_str(), a.start_line).cmp(&(b.path.as_str(), b.start_line)));
    for (i, c) in selected.iter_mut().enumerate() {
        c.alias = i;
    }
    Ok(selected)
}
