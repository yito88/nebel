use std::collections::HashMap;

use crate::metadata::{DocOrd, FieldId, MetadataValue};

// ---------------------------------------------------------------------------
// Filter expression
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum FilterExpr {
    And(Vec<FilterExpr>),
    Eq {
        field_id: FieldId,
        value: MetadataValue,
    },
    In {
        field_id: FieldId,
        values: Vec<MetadataValue>,
    },
    Range {
        field_id: FieldId,
        gte: Option<MetadataValue>,
        lte: Option<MetadataValue>,
    },
    Exists {
        field_id: FieldId,
    },
}

// ---------------------------------------------------------------------------
// Filter result
// ---------------------------------------------------------------------------

pub struct FilterResult {
    pub doc_ords: Vec<DocOrd>,
    pub matched_count: usize,
}

// ---------------------------------------------------------------------------
// Evaluation
// ---------------------------------------------------------------------------

/// Evaluate `expr` against a field_id-keyed metadata map.
/// Returns `true` if the document matches.
pub fn eval_expr(expr: &FilterExpr, fields: &HashMap<FieldId, MetadataValue>) -> bool {
    match expr {
        FilterExpr::And(exprs) => exprs.iter().all(|e| eval_expr(e, fields)),

        FilterExpr::Eq { field_id, value } => fields.get(field_id).is_some_and(|v| v == value),

        FilterExpr::In { field_id, values } => fields
            .get(field_id)
            .is_some_and(|v| values.iter().any(|val| v == val)),

        FilterExpr::Range { field_id, gte, lte } => match fields.get(field_id) {
            None => false,
            Some(v) => {
                let lower_ok = gte.as_ref().is_none_or(|lo| compare_values(v, lo) >= 0);
                let upper_ok = lte.as_ref().is_none_or(|hi| compare_values(v, hi) <= 0);
                lower_ok && upper_ok
            }
        },

        FilterExpr::Exists { field_id } => fields.contains_key(field_id),
    }
}

/// Numeric/string ordering for Range evaluation.
/// Panics if types mismatch — callers should validate filter types against
/// schema before evaluating.  Returns 0 for incomparable types (defensive).
fn compare_values(a: &MetadataValue, b: &MetadataValue) -> i32 {
    match (a, b) {
        (MetadataValue::Int64(x), MetadataValue::Int64(y)) => x.cmp(y) as i32,
        (MetadataValue::Float64(x), MetadataValue::Float64(y)) => {
            x.partial_cmp(y).map(|o| o as i32).unwrap_or(0)
        }
        (MetadataValue::Timestamp(x), MetadataValue::Timestamp(y)) => x.cmp(y) as i32,
        (MetadataValue::String(x), MetadataValue::String(y)) => x.cmp(y) as i32,
        // Type mismatch in range — no match.
        _ => i32::MAX,
    }
}
