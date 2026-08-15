"""Evaluate the nine authenticated frozen controllers on canonical MCD eval data.

Run mode writes one subject-disjoint shard.  Merge mode reloads the sources and
all checkpoints, reruns every rollout, and publishes only the recomputed rows.
"""
from __future__ import annotations
import argparse, csv, hashlib, json, os, shlex, sys
from importlib import metadata as package_metadata
from pathlib import Path

from adaptive_roi_rppg.contracts import canonical_json_bytes, sha256_file
from adaptive_roi_rppg.contracts.errors import ContractValidationError
from adaptive_roi_rppg.data.mcd import load_mcd_manifest_tree, read_mcd_canonical_frames
from adaptive_roi_rppg.evaluation.adapters.sb3_recurrent import load_frozen_recurrent_policy
from adaptive_roi_rppg.evaluation.model_plan import EXCLUSIONS, load_frozen_model_plan
from adaptive_roi_rppg.evaluation.model_replay import paired_subject_bootstrap, rollout_frozen_policy, score_frozen_rollout, summarize_model_rows
from adaptive_roi_rppg.evaluation.model_publication import CLIP_FIELDS, HOP_FIELDS, SUBJECT_FIELDS, csv_bytes as strict_csv_bytes, read_strict_csv
from adaptive_roi_rppg.labels.mcd import read_mcd_eval_labels
from adaptive_roi_rppg.signal import build_pos_measurements

SCHEMA="gate8-mcd-frozen-models-v1"; MARKERS={"started":"STARTED.json","failed":"FAILED.json","complete":"COMPLETE.json"}
def _fail(message): raise ContractValidationError(message)
def _json(path, value, *, exclusive=True):
    data=json.dumps(value,sort_keys=True,separators=(",",":"))
    mode="x" if exclusive else "w"
    with Path(path).open(mode,encoding="utf-8") as h: h.write(data)
def _marker(root,state,common,**extra): _json(root/MARKERS[state],{"state":state,**common,**extra})
def _sha(path): return sha256_file(path)
def _csv_bytes(rows):
    if not rows: _fail("cannot publish empty CSV")
    fields=tuple(rows[0])
    if any(tuple(row)!=fields for row in rows): _fail("CSV rows have inconsistent fields")
    expected = HOP_FIELDS if fields == HOP_FIELDS else CLIP_FIELDS if fields == CLIP_FIELDS else SUBJECT_FIELDS if fields == SUBJECT_FIELDS else None
    if expected is None: _fail("Gate 8 attempted to publish an unfrozen CSV schema")
    return strict_csv_bytes(rows, expected)
def _write_csv(path, rows):
    with Path(path).open("xb") as h: h.write(_csv_bytes(rows))

def _tree_hash(base):
    base=Path(base).resolve(); ignored={".git",".venv","__pycache__"}; files=[]
    def walk(path):
        for entry in sorted(path.iterdir(),key=lambda p:p.name.encode()):
            if entry.name in ignored: continue
            if entry.is_symlink(): _fail("code snapshot has a symlink")
            if entry.is_dir(): walk(entry)
            elif entry.is_file() and entry.suffix != ".pyc": files.append(entry.relative_to(base))
            else: _fail("code snapshot has an unsupported entry")
    walk(base); digest=hashlib.sha256()
    for relative in files: digest.update(relative.as_posix().encode()+b"\0"); digest.update((base/relative).read_bytes()+b"\0")
    return digest.hexdigest()

def _inventory(args, plan):
    paths=[Path(args.manifest_tree)/name for name in ("dataset_manifest.json","split_manifest.json","source_inventory.json")]+[Path(args.plan)]
    paths += [Path(spec["locator"]) for spec in plan.checkpoints]
    if any(path.is_symlink() or not path.is_file() for path in paths): _fail("source/checkpoint inventory path is invalid")
    files=[{"path":str(path),"sha256":_sha(path),"byte_size":path.stat().st_size} for path in sorted(set(paths),key=str)]
    runtime={name:package_metadata.version(name) if name in {dist.metadata["Name"] for dist in package_metadata.distributions()} else "not-installed" for name in ("numpy","scipy","stable-baselines3","sb3-contrib","gymnasium","torch","cloudpickle")}
    source={"files":files}; environment={"runtime":runtime}
    payload={"source":source,"environment":environment}
    return payload,hashlib.sha256(canonical_json_bytes(source)).hexdigest(),hashlib.sha256(canonical_json_bytes(environment)).hexdigest()

def _read_csv_rows(path):
    return read_strict_csv(path, HOP_FIELDS)
    # Kept below only as documentation of the former permissive parser.
    ints={"hop_idx","seed","proposed_action","executed_action","pre_hold_count","post_hold_count"}; floats={"hop_time_s","gt_hr_bpm","abs_error_bpm","selected_hr_bpm","selected_confidence","selected_ppr","selected_coverage","pre_belief_hr_bpm","post_belief_hr_bpm","post_belief_velocity","post_belief_std_bpm"}; booleans={"legal","selected_valid"}; optional={"previous_action","selected_hr_bpm","selected_coverage","selected_invalid_reason","override_reason"}
    with Path(path).open(newline="",encoding="utf-8") as h: raw=list(csv.DictReader(h))
    result=[]
    for row in raw:
        value={}
        for key,item in row.items():
            if key in optional and item=="": value[key]=None
            elif key in ints: value[key]=int(item)
            elif key in floats: value[key]=float(item)
            elif key in booleans: value[key]=item=="True"
            else: value[key]=item
        result.append(value)
    return result

def _args():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--merge",action="store_true"); p.add_argument("--shard-dir",action="append")
    p.add_argument("--output-dir",required=True); p.add_argument("--manifest-tree",required=True); p.add_argument("--state-root",required=True); p.add_argument("--gt-root",required=True); p.add_argument("--plan",required=True); p.add_argument("--shard-index",type=int,default=0); p.add_argument("--shard-count",type=int,default=1); p.add_argument("--code-snapshot-sha256",required=True)
    return p.parse_args()

def _bindings(args):
    bundle=load_mcd_manifest_tree(args.manifest_tree); ids=set(bundle.split_manifest.eval_clip_ids)
    if len(ids)!=540: _fail("Gate 8 requires the original 540-clip eval inventory before exclusions")
    clips=tuple(sorted((c for c in bundle.clip_manifests if c.clip_id in ids and c.clip_id not in EXCLUSIONS),key=lambda c:c.clip_id))
    if len(clips)!=533 or len({c.subject_id for c in clips})!=89 or set(c.clip_id for c in clips)&EXCLUSIONS or ids-set(c.clip_id for c in clips)!=EXCLUSIONS: _fail("Gate 8 requires exactly the Gate 6 533-clip/89-subject cohort")
    if {c.subject_id for c in clips}&{c.subject_id for c in bundle.clip_manifests if c.clip_id in bundle.split_manifest.train_clip_ids}: _fail("train/evaluation subject overlap")
    expected=sum(max(0,(c.state_row_count-round(8*c.camera_fps))//round(c.camera_fps)+1) for c in clips)
    if expected!=91227: _fail("Gate 8 expected hop cohort is not 91,227")
    return bundle,clips

def _evaluate(args, clips):
    from adaptive_roi_rppg.evaluation.model_replay import CheckpointIdentity
    class _FullFace:
        identity=CheckpointIdentity("full_face","fixed_full_face",None,None)
        def initial_state(self): return None
        def predict(self, observation, recurrent_state, *, episode_start): return 0, None
    plan=load_frozen_model_plan(args.plan); policies=[_FullFace(),*[load_frozen_recurrent_policy(spec) for spec in plan.checkpoints]]; bundle,_=_bindings(args); rows=[]
    for clip in clips:
        frames=build_pos_measurements(read_mcd_canonical_frames(bundle,args.state_root,clip.clip_id,"eval")); labels=read_mcd_eval_labels(bundle,args.gt_root,clip.clip_id,clip)
        expected=max(0,(clip.state_row_count-round(8*clip.camera_fps))//round(clip.camera_fps)+1)
        if len(frames)!=expected or len(labels)!=expected: _fail("canonical source does not match expected hop coverage")
        for policy in policies: rows.extend(score_frozen_rollout(rollout_frozen_policy(frames,policy,clip_id=clip.clip_id),labels,{"subject_id":clip.subject_id,"view":clip.view,"condition":clip.condition}))
    return sorted(rows,key=lambda r:(r["method_id"],r["clip_id"],r["hop_idx"]))

def _common(args,plan):
    if len(args.code_snapshot_sha256)!=64 or set(args.code_snapshot_sha256)-set("0123456789abcdef"): _fail("code snapshot hash must be lowercase SHA-256")
    job,node=os.environ.get("SLURM_JOB_ID"),os.environ.get("SLURMD_NODENAME")
    if not job or not node: _fail("Gate 8 requires Slurm")
    code_root=Path(__file__).resolve().parents[1]
    if _tree_hash(code_root)!=args.code_snapshot_sha256: _fail("declared code snapshot hash does not match runtime source")
    inventory, source_hash, environment_hash=_inventory(args,plan)
    return {"schema":SCHEMA,"plan_id":plan.plan_id,"code_snapshot_sha256":args.code_snapshot_sha256,"source_inventory_sha256":source_hash,"environment_sha256":environment_hash,"source_inventory":inventory,"job_id":job,"node":node,"command":shlex.join(sys.argv)}

def _publish_shard(args, plan, clips):
    root=Path(args.output_dir); common=_common(args,plan)|{"shard_index":args.shard_index,"shard_count":args.shard_count}
    if root.exists() or root.is_symlink() or not root.parent.is_dir(): _fail("shard output must be a fresh path under an existing real parent")
    root.mkdir(); _marker(root,"started",common)
    try:
        rows=_evaluate(args,clips); summary=summarize_model_rows(rows)
        payload={**common,"clip_ids":[c.clip_id for c in clips],"subject_ids":sorted({c.subject_id for c in clips}),"plan_payload":plan.payload,"summary":summary["checkpoint_results"]}
        _write_csv(root/"per_hop.csv",rows); _write_csv(root/"per_clip.csv",summary["clip_rows"]); _json(root/"shard_manifest.json",payload)
        side={n:_sha(root/n) for n in ("per_hop.csv","per_clip.csv","shard_manifest.json")}; _json(root/"artifacts.sha256",side)
        # Reread every fallible artifact before COMPLETE.
        if any(_sha(root/n)!=d for n,d in side.items()) or set(path.name for path in root.iterdir()) != {"STARTED.json","per_hop.csv","per_clip.csv","shard_manifest.json","artifacts.sha256"}: _fail("shard artifact changed before completion")
        _marker(root,"complete",common,sha256=_sha(root/"shard_manifest.json"))
    except BaseException:
        if not (root/MARKERS["complete"]).exists():
            try: _marker(root,"failed",common)
            except BaseException: pass
        raise

def _merge(args, plan, clips):
    root=Path(args.output_dir); common=_common(args,plan)
    if root.exists() or root.is_symlink() or not root.parent.is_dir(): _fail("merge output must be a fresh path under an existing real parent")
    if not args.shard_dir or len(args.shard_dir)!=args.shard_count: _fail("merge requires every deterministic shard")
    root.mkdir(); _marker(root,"started",common)
    try:
        seen=set()
        for i,name in enumerate(args.shard_dir):
            d=Path(name); manifest=json.loads((d/"shard_manifest.json").read_text()); complete=json.loads((d/"COMPLETE.json").read_text())
            started=json.loads((d/"STARTED.json").read_text())
            side=json.loads((d/"artifacts.sha256").read_text())
            provenance=("schema","plan_id","code_snapshot_sha256","source_inventory_sha256","environment_sha256","job_id","node","command","shard_index","shard_count")
            if set(path.name for path in d.iterdir()) != {"STARTED.json","per_hop.csv","per_clip.csv","shard_manifest.json","artifacts.sha256","COMPLETE.json"} or manifest.get("plan_payload")!=plan.payload or manifest.get("shard_index")!=i or manifest.get("shard_count")!=args.shard_count or complete.get("state")!="complete" or started.get("state")!="started" or complete.get("sha256")!=_sha(d/"shard_manifest.json") or any(manifest.get(k)!=started.get(k) or complete.get(k)!=started.get(k) for k in provenance) or any(manifest.get(k)!=common.get(k) for k in ("schema","plan_id","code_snapshot_sha256","source_inventory_sha256","environment_sha256")) or (d/"FAILED.json").exists() or set(side)!={"per_hop.csv","per_clip.csv","shard_manifest.json"} or any(_sha(d/n)!=v for n,v in side.items()): _fail("shard terminal/provenance state is invalid")
            subjects=set(manifest["subject_ids"])
            if seen&subjects: _fail("subject appears in more than one shard")
            seen|=subjects
            shard_clips=tuple(c for c in clips if c.subject_id in subjects)
            expected=_evaluate(args,shard_clips)
            stored=_read_csv_rows(d/"per_hop.csv")
            if (d/"per_hop.csv").read_bytes()!=_csv_bytes(expected) or _csv_bytes(summarize_model_rows(stored)["clip_rows"]) != (d/"per_clip.csv").read_bytes(): _fail("shard rows do not match source-authoritative replay")
        if seen!={c.subject_id for c in clips}: _fail("shards do not cover the canonical cohort")
        # Source-authoritative recomputation: the published CSVs come only from this fresh replay.
        rows=_evaluate(args,clips); summary=summarize_model_rows(rows)
        learned=[r for r in summary["clip_rows"] if r["method_id"]!="full_face"]
        if len(learned)!=4797 or len({r["subject_id"] for r in learned})!=89 or sum(r["hop_count"] for r in learned)!=91227*9 or {item["family"] for item in summary["family_results"]}!={"dagger","standard_ppo","advantage_ppo"}: _fail("merged learned cohort is incomplete")
        _write_csv(root/"per_hop.csv",rows); _write_csv(root/"per_clip.csv",summary["clip_rows"]); _write_csv(root/"per_subject.csv",summary["subject_rows"])
        comparisons=[]
        for seed in range(3):
            for pair, seed_value in (((f"dagger_seed{seed}","full_face"),8101+seed),((f"standard_ppo_seed{seed}",f"dagger_seed{seed}"),8111+seed),((f"advantage_ppo_seed{seed}",f"dagger_seed{seed}"),8121+seed),((f"advantage_ppo_seed{seed}",f"standard_ppo_seed{seed}"),8131+seed)):
                comparisons.extend((paired_subject_bootstrap(summary["clip_rows"],*pair,estimand="equal_clip",seed=seed_value),paired_subject_bootstrap(summary["clip_rows"],*pair,estimand="equal_subject",seed=seed_value)))
        report={**common,"cohort":{"original_eval_clip_count":540,"clip_count":533,"subject_count":89,"expected_hops_per_checkpoint":91227,"exclusions":sorted(EXCLUSIONS)},"checkpoint_results":summary["checkpoint_results"],"family_results":summary["family_results"],"view_condition_rows":summary["view_condition_rows"],"paired_comparisons":comparisons,"limitations":["Frozen MCD evaluation only; no training or MMPD access.","Historical and canonical POS pipelines are not identical.","Checkpoint training-config provenance is descriptive_unverified; file/hash/architecture identity is verified."],"plan_payload":plan.payload}
        run_manifest={**common,"status":"complete","checkpoint_identities":[{k:spec[k] for k in ("method_id","family","seed","sha256","byte_size","provenance_status","training_config_provenance")} for spec in plan.checkpoints],"cohort":report["cohort"]}
        _json(root/"report.json",report); _json(root/"run_manifest.json",run_manifest)
        side={n:_sha(root/n) for n in ("per_hop.csv","per_clip.csv","per_subject.csv","report.json","run_manifest.json")}; _json(root/"artifacts.sha256",side)
        reread=summarize_model_rows(_read_csv_rows(root/"per_hop.csv"))
        if _csv_bytes(reread["clip_rows"])!=(root/"per_clip.csv").read_bytes() or _csv_bytes(reread["subject_rows"])!=(root/"per_subject.csv").read_bytes() or reread["checkpoint_results"]!=report["checkpoint_results"] or set(path.name for path in root.iterdir()) != {"STARTED.json","per_hop.csv","per_clip.csv","per_subject.csv","report.json","run_manifest.json","artifacts.sha256"} or any(_sha(root/n)!=d for n,d in side.items()): _fail("merged artifact changed or aggregation disagrees before completion")
        _marker(root,"complete",common,sha256=_sha(root/"report.json"))
    except BaseException:
        if not (root/MARKERS["complete"]).exists():
            try: _marker(root,"failed",common)
            except BaseException: pass
        raise

def main():
    args=_args(); plan=load_frozen_model_plan(args.plan); _,clips=_bindings(args)
    if args.shard_count<1 or not 0<=args.shard_index<args.shard_count: _fail("invalid shard assignment")
    if args.merge: _merge(args,plan,clips)
    else:
        subjects=sorted({c.subject_id for c in clips}); selected=set(subjects[args.shard_index::args.shard_count]); _publish_shard(args,plan,tuple(c for c in clips if c.subject_id in selected))
    return 0
if __name__=="__main__": raise SystemExit(main())
