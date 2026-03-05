/// A lightweight in-memory representation of a source file.
///
/// This type is intentionally minimal to keep `intelligence` pure and reusable.
/// Future phases can add caching/incremental update via a `DocumentStore`.

pub fn build_line_end_indices(content: &str) -> Vec<usize> {
    content
        .as_bytes()
        .iter()
        .enumerate()
        .filter_map(|(i, &b)| (b == b'\n').then_some(i))
        .collect()
}
