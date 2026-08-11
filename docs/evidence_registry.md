# Evidence registry

This document solely owns evidence paths, hashes, sizes, artifact mtimes, cohorts, rulers, aggregation, uncertainty, and supersession. Local numerical values are from the supplied frozen packet; remote values are not locally recomputed. Frozen evidence packet generation time: `2026-08-11T03:11:11.981128+00:00`.

## Method definitions

- **DAgger base:** Architecture LSTM controller; learning algorithm initial cross-entropy cloning of Oracle-C followed by two self-rollout rounds with greedy one-step level-B relabeling and refitting; initialization random/from-scratch; ensemble/gate single model per seed with no deployment ensemble or fallback gate; online interactive imitation collection followed by offline frozen self-rollout evaluation.
- **Advantage PPO:** Architecture LSTM controller; learning algorithm online PPO with advantage-over-base reward shaping; initialization warm-start from the matching DAgger base; ensemble/gate single model per seed with no deployment ensemble or fallback gate; online on-policy MCD training followed by offline frozen self-rollout evaluation.
- **E-009 clone arms:** K8-clone and selected-teacher clone are LSTM controllers trained by supervised cross-entropy cloning from the respective GT-informed offline teacher labels; each uses random/from-scratch initialization, a single model with no deployment ensemble or gate, and offline supervised training followed by frozen causal self-rollout evaluation.
- **Oracle-C:** Explicitly non-learned, offline, GT-informed, and non-deployable; it is a beam-search diagnostic rather than an inference-time controller.

## Claim registry

| ID | Claim or question | Status | Dataset / split / cohort | Ruler and aggregation | Primary row-level artifact | SHA-256 | Size bytes | Artifact mtime UTC | Supporting manifest or method source | Locally recomputed value | Uncertainty or required control | Limit or supersession |
|---|---|---|---|---|---|---|---:|---|---|---|---|---|
| E-001 | DAgger base | FACT | MCD test90; 533 clips, 89 subjects | schema-v3 no-subharmonic causal; mean `mae_clip` | three base CSVs below | see provenance | see provenance | 2026-07-10 UTC | replay manifest E-015 | 7.6044 / 7.9429 / 8.7998; mean 8.1157 BPM | 3 seeds; subject pairing E-003 | none |
| E-002 | Advantage PPO | FACT | MCD test90; 533 clips, 89 subjects | same ruler; mean `mae_clip` | three PPO CSVs | see provenance | see provenance | 2026-07-10 UTC | replay manifest E-015 | 7.3318 / 7.2135 / 7.5792; mean 7.3748 BPM | 3 seeds; subject pairing E-003 | none |
| E-003 | Base minus PPO | FACT | MCD test90; 533 clips, 89 subjects | paired per-clip evaluation; subject-block bootstrap | `subject_block_bootstrap.json` | `7e9e0c6dac161a5462f694b0cdacd9bd5f86de6c73aadd88cd1bd3ab676cc2bb` | 778 | 2026-07-10 04:48 UTC | B=10,000; seed 20260709; unit subject | mean 0.7409 BPM; per-seed 0.2726 / 0.7293 / 1.2206 | 95% CI [0.5364, 0.9453] BPM | positive means PPO improved |
| E-004 | Full-face level A | FACT | MCD test90; 533 clips, 89 subjects | schema-v3 phase-0 causal; mean `mae_clip` | `phase0_v3nc/per_clip.csv` | `7481c4e5bcecda77dd954e78d49d7e6cb435a5353bf89f6a4a30187ce2169f2e` | 236353 | 2026-07-10 05:34 UTC | phase-0 artifact | 12.3865 BPM; 533 unique stems; 91,227 valid hops | row-level recomputation required | active full-face reference |
| E-005 | Oracle-C K=8 and wider-search limit | FACT with LIMITATION | MCD test90; 533 clips | offline GT-informed beam; mean per-clip MAE | `phase0_v3nc/per_clip.csv`; `beam_k2048.csv` | see provenance | see provenance | 2026-07-10 / 2026-07-14 UTC | Oracle-C manifests | K=8 2.1084; K=2048 1.6872 BPM | not inference-time; not optimal | K=8 frozen historical reference |
| E-006 | PPO versus greedy-B behavior | FACT diagnostic | MCD; 178,446 decision hops | same-state free-choice replay | disagreement and rank CSVs | see provenance | see provenance | 2026-07-10 17:59 UTC | rank manifest | disagreement 69.2252%; rank1 31.1635%; rank12 18.5670% | denominators recorded; diagnostic only | no causal mechanism claim |
| E-007 | Critic recovery explanation | NEGATIVE RESULT | MCD schema-v3 audit | cross-validated recovery prediction | critic audit CSV/JSON | see provenance | see provenance | 2026-07-22 UTC | audit controls | no meaningful added recovery prediction | audited explanation only | not all critics/PPO |
| E-008 | Retired schema-v2 headline | SUPERSEDED | MCD schema-v2 | subharmonic-on ruler | `phase0_v3nc/cache_manifest_v3.json` | `d9c63ab03582ae254e067ce103e6ddb563019c65f1087ee723bcd9614d55aadf` | 2826 | 2026-07-10 05:34 UTC | cache manifest | no current value | use E-001–E-003 | 7.44 BPM retired |
| E-009 | Wider-search teacher transfer | NEGATIVE RESULT | MCD test90 clone | causal self-rollout; per-clip MAE | K=8 and selected self-rollout CSVs | see provenance | see provenance | 2026-07-14 23:17 UTC | clone gate summary/bootstrap | K=8 15.3535; selected 15.5924 BPM | stop before DAgger/PPO | selected teacher failed gate |
| E-010 | Official POS MMPD | REMOTE-ONLY | MMPD; 300 clips | official POS_WANG/FFT; remote per-clip MAE | `Polaris: ~/results/mmpd_official_pos_20260807_full300/per_clip.csv` | unavailable locally | unavailable locally | remote run documented 2026-08-08 | `/Users/atharva/Desktop/RL for RoI/docs/mmpd_official_pos_comparison_2026-08-08.md` | 15.8174 BPM documented; not locally recomputed | transfer only | separate ruler |
| E-011 | Unchanged frozen-controller predictions rescored under MMPD-specific corrected-GT rule; TS-CAN comparison | REMOTE-ONLY and INCOMPLETE locally | MMPD transfer | corrected-GT causal comparison | `Polaris: ~/RL-for-RoI/results/mmpd_corrected_comparison_20260804_run15/` | unavailable locally | unavailable locally | remote run documented 2026-08-04 | `/Users/atharva/Desktop/RL for RoI/docs/mmpd_corrected_tscan_comparison_2026-08-04.md` | no locally verified value | exact row-level package missing | no MCD feedback |
| E-012 | Official versus causal POS limitation | LIMITATION | cross-dataset ruler comparison | crop, filtering, windows, aggregation differ | official POS comparison note | see provenance | see provenance | 2026-08-08 UTC | same note | no value | not a smoothing ablation | keep rulers separate |
| E-013 | Standard PPO transfer | REMOTE-ONLY | MMPD frozen transfer | frozen MMPD evaluation | MMPD evaluation record | see provenance | see provenance | 2026-08-04 UTC | `/Users/atharva/Desktop/RL for RoI/docs/mmpd_eval_record_2026-07-30.md` | no locally verified value | no development use | remote-only |
| E-014 | Golden/fitted-Q line | HISTORICAL CONTEXT | legacy golden per-window | not comparable to streaming | streamlined report | see provenance | see provenance | 2026-06-12 UTC | report | no scoreboard here | navigation only | not active evidence |
| E-015 | MCD training provenance | FACT with LIMITATION | 510 train subjects; 89 test subjects | disjoint identities; separate 3,057-clip manifest | train list; Oracle-C manifest; replay manifest | see provenance | see provenance | 2026-06-12 to 2026-08-10 UTC | replay manifest and checkpoint records | overlap 0; train substrate 3,057 clips | exact checkpoint-consumed rows unproven | manifest-backed checkpoint identity |

## Local artifact provenance

| Artifact | SHA-256 | Bytes | Artifact mtime UTC |
|---|---|---:|---|
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/ppo_per_clip_seed0_base.csv` | `dda0cb5b7101f2f7a861a515cb24f1f01292924fe227d0b9d3a36741c6dd0a54` | 32682 | 2026-07-10 04:41 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/ppo_per_clip_seed1_base.csv` | `999ac87768ab544ed972e0335f0ebbb1243f3ec835b3f3c90df42bf39d6fd317` | 32715 | 2026-07-10 04:42 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/ppo_per_clip_seed2_base.csv` | `907f6117d5134094c38901559436bcbe65e223fcd3ace9973a8c5128977af961` | 32710 | 2026-07-10 04:43 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/ppo_per_clip_seed0_ppo.csv` | `46cc34c92a97a75532aecbe013555bb88fd31fdb52038480bae1ab0b165b0a12` | 32699 | 2026-07-10 04:44 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/ppo_per_clip_seed1_ppo.csv` | `1807b177b1ab0ffa73da8f63bf4fb551561ab9d916ae44c26cf7fdff5ec97ed3` | 32728 | 2026-07-10 04:46 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/ppo_per_clip_seed2_ppo.csv` | `b0b3218aab0144b9b15226b2f124831aa983608affe1e71fdcaa50f43bac5c3f` | 32690 | 2026-07-10 04:47 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/subject_block_bootstrap.json` | `7e9e0c6dac161a5462f694b0cdacd9bd5f86de6c73aadd88cd1bd3ab676cc2bb` | 778 | 2026-07-10T04:48:03.220644+00:00 |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/phase0_v3nc/per_clip.csv` | `7481c4e5bcecda77dd954e78d49d7e6cb435a5353bf89f6a4a30187ce2169f2e` | 236353 | 2026-07-10 05:34 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/phase0_v3nc/cache_manifest_v3.json` | `d9c63ab03582ae254e067ce103e6ddb563019c65f1087ee723bcd9614d55aadf` | 2826 | 2026-07-10T05:34:31.098554+00:00 |
| `/Users/atharva/Desktop/RL for RoI/configs/reproducibility/streaming_v3nc_replay.json` | `b176b09fbb2e169531ce10f817eb284568a7b0b191f89c8a7a8aaa624a87169d` | 6441 | 2026-08-10 04:10 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/splits/streaming_train_disjoint.csvlist.txt` | `1e0e1e281d8b273a6689de426dbdf39e46b74a9881c340c23b60e1dd0b93896f` | 2550 | 2026-06-12 04:28 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/oracle_c_v2_all_2026-07-13_v3nc/manifest.json` | `2ff143da4f329b7372189f74c4500bc79ff4fa9489026f5ddb1e08f7dcb96c07` | 9298 | 2026-07-14 23:19 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/oracle_c_validation_2026-07-12_v3nc/fullclip_beam_shards_v2/beam_k2048.csv` | `545dcada04424ed300ca2f775188ecd9e41c8a5940928e7715fe216461979d34` | 338950 | 2026-07-14 02:28 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/oracle_c_v2_all_2026-07-13_v3nc/beam_width_summary.csv` | `0951b0a20d374a454d4340180a4e69f2a31cdcaf8416b61d453e8cdd59590ca2` | 1054 | 2026-07-14 23:19 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/rl_vs_greedyb_regret_analysis/disagreement_regret_3seed_pooled.csv` | `7917c0a7681f55cfc96e2ee8ac869b2e9ecede2136d40362f9ffafb55c61c60a` | 1955 | 2026-07-10 17:59 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/rl_vs_greedyb_rank_analysis/agent_choice_rank_distribution.csv` | `53868ece862ff75247dd7c039e1d55fdb7bbbc2a2a4179b001ab59537c39511c` | 14198 | 2026-07-10 17:59 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/rl_vs_greedyb_rank_analysis/manifest.json` | `8c65676eff9288e9ab839e658594a19ac3116cd5fc87518775fb1ec15a4b7c29` | 240 | 2026-07-10 17:59 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/critic_recovery_audit/cross_validated_prediction.csv` | `d2c61d79dfd308087389250a28d817fcfd52affa5794e3cb4d95c7c5fa6c32e4` | 673 | 2026-07-22 07:06 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/critic_recovery_audit/subject_block_bootstrap.json` | `1c65ea61840bc3d88dca5a66c482a3d7c580823d477064504f26416586b48e29` | 1004 | 2026-07-22 07:58 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/final_eval_2026-07-09_v3nc/critic_recovery_audit/negative_control.json` | `042469c8a0ac67c602688cb625c8b5978193bc682db4e95c6dbf08c340330241` | 1165 | 2026-07-22 07:06 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/oracle_c_v2_clone_ab_2026-07-14_v3nc/k8/test90_self_rollout_per_clip.csv` | `cf3e3f0f75a5a9195a4fce3a670085445ea5413c51023011e41304f07aeb40b7` | 232822 | 2026-07-14 23:17 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/oracle_c_v2_clone_ab_2026-07-14_v3nc/selected/test90_self_rollout_per_clip.csv` | `cd0ad1c6ff9c0109f073704a8a77a22313585a72386b281aa6b2ef7ab0ce0c12` | 226759 | 2026-07-14 23:17 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/oracle_c_v2_clone_ab_2026-07-14_v3nc/clone_gate_summary.csv` | `fad604465e48b66bdabf4da1c8f6dbd3adef7fe201d297f94b2e246d41497624` | 326 | 2026-07-14 23:17 UTC |
| `/Users/atharva/Desktop/RL for RoI/results/streaming/oracle_c_v2_clone_ab_2026-07-14_v3nc/selected_vs_k8_subject_bootstrap.json` | `5018f43b2018aef971a5afd470aae80e5f1414bc6e1bcb1c61b781ab1210dfda` | 255 | 2026-07-14 23:17 UTC |
| `/Users/atharva/Desktop/RL for RoI/docs/mmpd_official_pos_comparison_2026-08-08.md` | `a1941c2024c6608cba70cbdc3ff82598fb69f64cf76a3a3312aec869fa54bef9` | 5078 | 2026-08-08 17:59 UTC |
| `/Users/atharva/Desktop/RL for RoI/docs/mmpd_corrected_tscan_comparison_2026-08-04.md` | `2d29f6d23467e9a5640c8c41417ee58721ebb002b88a9beff68f47371ef42350` | 5989 | 2026-08-04 21:26 UTC |
| `/Users/atharva/Desktop/RL for RoI/docs/mmpd_eval_record_2026-07-30.md` | `0b2bb4c0ab6290a6b99222e40f9af9dce1eaf19b5d232f11620615398de4d79e` | 24823 | 2026-08-04 08:01 UTC |
| `/Users/atharva/Desktop/RL for RoI/docs/golden_to_t69_streamlined_report.md` | `76f5b1d4434c32a84016c3613e9011ccc687c5d3413a090b27de9884fdd601de` | 4988 | 2026-06-12 03:07 UTC |

Remote artifacts are intentionally listed only by `Polaris:` path in the claim table; their local size and hash are unavailable.

Manifest-backed checkpoint identities (not directly hashed locally):

| Filename | Role / seed | SHA-256 | Bytes |
|---|---|---|---:|
| `ppo_seed0_dagger_r2_seed0_advbase_ppo2m_base.zip` | base / 0 | `4ced6433b30e45211ccf483324302a73c5fe25528f2bdf69dcd065ecec942d0d` | 1104456 |
| `ppo_seed1_dagger_r2_seed1_advbase_ppo2m_base.zip` | base / 1 | `350692f169e80b5c056d38ac9fd3ef9636d46bdaf90c59e3b99b2c3a383c7050` | 1104460 |
| `ppo_seed2_dagger_r2_seed2_advbase_ppo2m_base.zip` | base / 2 | `bc4b59ab1a6bb10983b192b51548981cedac2086bec4554d372d24a1d565f27c` | 1104456 |
| `ppo_seed0_dagger_r2_seed0_advbase_ppo2m.zip` | PPO / 0 | `5c48245700d0b985c212369e1398c83af8ab13b6182fc59e80b1ab299cc0c506` | 3296076 |
| `ppo_seed1_dagger_r2_seed1_advbase_ppo2m.zip` | PPO / 1 | `31f1e55131ba3918c0d853dd6a836b837f443a8addf3aba9ef3699829a9dc31a` | 3296080 |
| `ppo_seed2_dagger_r2_seed2_advbase_ppo2m.zip` | PPO / 2 | `1d9a2893595bafef748e9062b224fdc5e1c3213060e28233955610498dd1fc35` | 3296076 |

## Supersession index

| Earlier item | Current interpretation |
|---|---|
| Schema-v2 7.44 | Superseded by schema-v3 7.375 comparison (E-008 → E-001–E-003). |
| MMPD job 46340 / 241 clips | Superseded by job 46352 / 300 clips (E-010). |
| July MMPD comparison | Superseded by the August corrected-GT comparison (E-011). |
| K=8 “optimal oracle” wording | Superseded by frozen historical K=8 reference plus wider-search correction (E-005). |
| Legacy golden direct-control headline | Historical context only; active question is streaming causal control (E-014). |
