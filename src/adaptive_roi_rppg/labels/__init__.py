"""MCD-only ground-truth label adapters."""

from .mcd import GT_RULE_ID, GT_RULE_PAYLOAD, GT_RULE_PAYLOAD_SHA256, read_mcd_labels

__all__ = ["GT_RULE_ID", "GT_RULE_PAYLOAD", "GT_RULE_PAYLOAD_SHA256", "read_mcd_labels"]
