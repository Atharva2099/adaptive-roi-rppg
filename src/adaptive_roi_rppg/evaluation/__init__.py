"""Frozen, framework-neutral evaluation and immutable publication."""

from .core import (
    CLIP_FIELDS, HOP_FIELDS, SUBJECT_FIELDS, ClipBinding, FrozenEvaluationPlan, RunProvenance,
    build_train_full_face_plan, evaluate, evaluate_and_publish, publish_evaluation, verify_publication,
    verify_publication_against_sources, verify_publication_structure,
)
from .parity import (
    HistoricalCacheBinding, Gate6ParityPlan, HistoricalFullFaceHop, ParityHopRow,
    ParityClipRow, ParitySummary, load_historical_cache_inventory,
    build_gate6_fixture_plan, build_gate6_full_plan, replay_historical_full_face,
    evaluate_current_full_face, compare_full_face_trajectories, classify_discrepancy,
    summarize_gate6_rows, publish_gate6_report, verify_gate6_publication,
)
from .crossover import (
    ACTION_COUNT, ARM_IDS, HISTORICAL_ORACLE_STATUS, LEARNED_ARM_STATUS,
    aggregate_crossover, build_crossover_rows, merge_crossover_shards,
    publish_crossover_shard, score_current_arm, score_historical_arm,
)
from .failure_audit import (
    ACTION_COUNT as FAILURE_AUDIT_ACTION_COUNT,
    ScoredCounterfactual, ScoredFailureAudit, UnscoredCounterfactual,
    UnscoredFailureAudit, build_failure_audit, random_legal_requested_action,
    score_failure_audit,
)

__all__ = ["CLIP_FIELDS", "HOP_FIELDS", "SUBJECT_FIELDS", "ClipBinding", "FrozenEvaluationPlan", "RunProvenance", "build_train_full_face_plan", "evaluate", "evaluate_and_publish", "publish_evaluation", "verify_publication_structure", "verify_publication_against_sources", "verify_publication", "HistoricalCacheBinding", "Gate6ParityPlan", "HistoricalFullFaceHop", "ParityHopRow", "ParityClipRow", "ParitySummary", "load_historical_cache_inventory", "build_gate6_fixture_plan", "build_gate6_full_plan", "replay_historical_full_face", "evaluate_current_full_face", "compare_full_face_trajectories", "classify_discrepancy", "summarize_gate6_rows", "publish_gate6_report", "verify_gate6_publication", "ACTION_COUNT", "ARM_IDS", "HISTORICAL_ORACLE_STATUS", "LEARNED_ARM_STATUS", "aggregate_crossover", "build_crossover_rows", "merge_crossover_shards", "publish_crossover_shard", "score_current_arm", "score_historical_arm", "FAILURE_AUDIT_ACTION_COUNT", "ScoredCounterfactual", "ScoredFailureAudit", "UnscoredCounterfactual", "UnscoredFailureAudit", "build_failure_audit", "random_legal_requested_action", "score_failure_audit"]
