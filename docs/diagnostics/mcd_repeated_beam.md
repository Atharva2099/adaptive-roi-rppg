# MCD repeated-correction beam diagnostic

Question: does an offline repeated correction trajectory provide headroom over factual PPO actions on the full frozen MCD evaluation cohort?

The required arms are factual PPO replay, the one-time intervention arm (each legal action forced once at an eligible anchor), and the repeated beam arm. The one-time arm is the comparison needed to distinguish a repeated search effect from a single correction. A synthetic controller test is the negative control for branch isolation, action coverage, and recurrent-state copying.

This is an offline ground-truth diagnostic. The rank-1 action uses immediate post-update ground-truth error, and retained paths are ordered by cumulative error, endpoint error, executed sequence, and origin sequence. It is headroom evidence, not a deployable policy result. Analyzer outputs contain final survivors only and cannot reconstruct intermediate beam winners. MMPD is not accessed, and no new numerical experiment has been executed by this integration.

Prepare the authenticated full-cohort subsets:

```sh
PYTHONPATH=src python3 scripts/prepare_mcd_repeated_beam_subsets.py \
  --source-csv PATH/per_hop_counterfactual.csv \
  --plan-json PATH/audit_plan.json \
  --complete-json PATH/COMPLETE.json \
  --output-dir PATH/subsets
```

Run one worker per task through the retained CPU-cluster launcher, then analyze completed outputs:

```sh
sbatch slurm/run_mcd_repeated_correction_beam_full_cpucluster.sbatch
PYTHONPATH=src python3 scripts/analyze_mcd_repeated_beam_behavior.py \
  --results-root PATH/results \
  --output-dir PATH/analysis \
  --job-id JOB_ID \
  --source-subsets-dir PATH/subsets \
  --beam-config configs/evaluation/mcd_repeated_correction_beam_full_v1.json
```

The producer and analyzer require the frozen MCD source-plan identities, model plan, checkpoint package, manifest tree, state root, and evaluation-label root supplied by the launcher or command-line options. Keep the full mapping `task_index = 3 * source_plan_clip_position + seed` and the 533-clip, 89-subject cohort fixed. Do not use historical worst-54 outputs as current evidence.
