# Gate 8: frozen MCD model evaluation

Status: `INCOMPLETE` / `NO-GO`. Gate 8 is not accepted. No model result, model-quality claim, or completion claim is recorded. No retraining was authorized, no MMPD material was used, and no Polaris fixture or full evaluation may run until the final Sol findings are repaired and reviewed.

## Approved scope

The approved plan was a frozen MCD evaluation of nine existing `RecurrentPPO` checkpoint files: three DAgger checkpoints, three Standard-PPO checkpoints, and three Advantage-PPO checkpoints. The planned cohort is 533 valid evaluation clips representing 89 subjects. The plan allows no retraining and no MMPD access. It requires row-level outputs, behavior information, provenance, deterministic shard ownership, and independently rebuilt publication artifacts.

## Evaluation history

All timestamps below are `2026-08-14 UTC` because exact times were not retained in the handoff.

### Sol approved plan — `2026-08-14 UTC`

Sol approved the frozen MCD-only evaluation plan described above. The intended output was an auditable comparison of the existing checkpoint files, not a new training result. The MMPD evaluation-only boundary remained unchanged.

### Terra first implementation — `2026-08-14 UTC`

Terra verified that all nine checkpoint files were present on Polaris and CPU-loaded successfully. The inspected models had a 101-value float32 observation, 12 actions, and a recurrent 128-unit LSTM. Terra reported a local focused test result of 6/6 and a passing full suite. No Polaris evaluation run was started.

This established checkpoint loading and basic interface evidence only. It did not establish model-training provenance, shard correctness, publication correctness, or an evaluation result.

### First Sol review — `NO-GO`, `2026-08-14 UTC`

Sol found these issues:

- The launcher `RUN` path did not propagate correctly. This could make the requested production execution differ from the tested path.
- Standard-PPO training provenance remained unresolved. A family name and checkpoint file are not enough to authenticate how the checkpoint was trained.
- The report and behavior outputs were incomplete. The planned evidence could not fully explain what each checkpoint did.
- The bootstrap estimand did not match the stated analysis. A confidence interval is not useful if its sampling unit and reported quantity differ.
- Merge and provenance validation were weak. A successful shard or merge could not by itself prove that the source inputs, ownership, and checkpoint identity were correct.
- Publication tests were insufficient. Important failure and tamper cases were not exercised.

Why this mattered: a CPU load smoke can pass while the actual launcher, statistical interpretation, provenance, or published result is wrong. The review therefore rejected any Polaris evaluation or Gate 8 acceptance.

### Terra first repair — `2026-08-14 UTC`

Terra attempted to repair the findings by fixing launcher propagation, marking uncertain training provenance as `descriptive_unverified`, expanding report and behavior metrics, separating bootstrap estimands, and adding a `run_manifest` with stronger hashes and coverage checks. Terra reported 12/12 focused tests and 122 passing tests in the full suite. No Polaris run was started.

These changes addressed the identified areas in the local implementation, but they did not constitute acceptance. The next Sol review still found publication and identity gaps.

### Second Sol review — `NO-GO`, `2026-08-14 UTC`

Sol found:

- The final report was not fully revalidated after the repair.
- Shard ownership and clip IDs were not strict enough to prove exact, non-overlapping coverage.
- Behavior fields were still incomplete for the intended analysis.
- The publication path still lacked direct tests.

Why this mattered: even stronger hashes and a manifest do not make a final report trustworthy if the report is not rebuilt and checked, if shard ownership is only trusted, or if the production publication path is untested.

### Sol repair plan — `2026-08-14 UTC`

Sol required the next repair to include:

- a dedicated `model_publication` module;
- exact marker and CSV schemas;
- deterministic subject stride and clip IDs;
- shard reaggregation and source replay;
- an independent final artifact rebuild;
- identity-preserving aggregates;
- real publication, merge, and tamper tests.

These were repair conditions, not an approval to run a fixture or full evaluation.

### Terra latest repair — `2026-08-14 UTC`

Terra added `model_publication.py`, expanded `model_replay.py`, integrated serialization changes, and added `test_gate8_model_publication.py`. Terra reported 127 total local tests and 68 focused Gate 5–8 tests passing. No Polaris evaluation, MMPD access, or legacy-code run occurred during this repair.

This is local engineering evidence only. Terra did not solve the Gate 8 acceptance issues, and the reported test counts do not prove that the production runner publishes or merges safely.

### Final Sol review — `NO-GO`, `2026-08-14 UTC`

The final review found the following unresolved issues:

- The strict publication helpers are not wired into the production runner. Production still uses permissive markers.
- Deterministic ownership is still trusted by the runner, while the helper rejects repeated subjects. The production path therefore does not prove the ownership rule it relies on.
- The report and `run_manifest` are not reread and rebuilt as final acceptance checks.
- Aggregation groups only by `method_id` and `clip_id`, which can erase checkpoint identity when multiple checkpoint files share those values.
- Tests still do not call the production publish and merge path. Helper-only tests cannot establish production behavior.
- The checkpoint manifest directly references `/Users/atharva/Desktop/RL for RoI` paths. That violates the intended clean-runtime boundary and leaves the clean repository dependent on the legacy checkout.

Why this mattered: these are integrity, identity, and reproducibility failures in the path that would create the published evidence. They prevent a reviewer from knowing that a published row belongs to the intended checkpoint, shard, source replay, and final report.

## Required next repair conditions

Gate 8 remains `INCOMPLETE` / `NO-GO` until all of the following are demonstrated in code review and local tests:

1. The production runner uses the strict publication helpers and strict marker/CSV schemas.
2. The runner computes and verifies deterministic subject ownership and exact clip IDs; repeated subjects and duplicate or missing clips fail closed.
3. Checkpoint identity is preserved through replay, shard outputs, merges, reports, and all aggregate keys.
4. The production runner rereads and independently rebuilds the final report and `run_manifest` from the published row-level artifacts.
5. Tests directly invoke production publication and merge, including malformed markers, duplicate ownership, missing/extra clips, tampered rows, source-replay mismatch, and checkpoint-identity collisions.
6. The checkpoint manifest uses clean-runtime references and no direct dependency on the legacy `RL for RoI` checkout.
7. Sol reviews the repaired diff and accepts it. Only after that review may a bounded Polaris fixture be considered; until then, no Polaris fixture or full evaluation is allowed.

## Evidence boundary

The checkpoint files and their architecture/interface inspection are not Gate 8 results. The local test counts are not a Polaris evaluation. The existing MCD cohort definition is inherited from the accepted earlier gate, but no Gate 8 model outputs, confidence intervals, behavior conclusions, or model comparison numbers are claimed here.
