use std::collections::HashSet;

use crate::types::SearchHit;

pub fn hits_to_ids(hits: &[SearchHit]) -> Vec<String> {
    hits.iter().map(|h| h.doc_id.clone()).collect()
}

pub fn recall_at_k(truth_ids: &[String], result_ids: &[String], k: usize) -> f64 {
    let truth_set: HashSet<&str> = truth_ids.iter().take(k).map(|s| s.as_str()).collect();
    let result_set: HashSet<&str> = result_ids.iter().take(k).map(|s| s.as_str()).collect();
    let intersection = truth_set.intersection(&result_set).count();
    intersection as f64 / k as f64
}

pub fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p / 100.0 * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}
