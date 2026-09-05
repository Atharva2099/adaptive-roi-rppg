import os
import subprocess
import tempfile
import unittest
from pathlib import Path


class FullCpuclusterLauncherTests(unittest.TestCase):
    def test_seed_mode_rejects_other_seed_or_partial_range(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/run_mcd_repeated_correction_beam_full_cpucluster.sbatch"
        with tempfile.TemporaryDirectory() as tmp:
            common = {
                "BASE": str(launcher.parents[1]), "OUT": str(Path(tmp) / "out"),
                "SOURCE_SUBSETS_DIR": str(Path(tmp) / "subsets"), "MCD_MANIFEST_TREE": str(Path(tmp) / "manifest"),
                "MCD_STATE_ROOT": str(Path(tmp) / "state"), "MCD_GT_ROOT": str(Path(tmp) / "gt"),
                "CHECKPOINT_ROOT": str(Path(tmp) / "checkpoints"), "PYTHON": "/bin/true",
                "SLURM_JOB_ID": "test", "PATH": "/usr/bin:/bin",
            }
            for seed, first, last in ((3, 0, 532), (1, 1, 532), (1, 0, 531)):
                env = {**common, "RUN_SEED": str(seed), "FIRST_CLIP_POSITION": str(first), "LAST_CLIP_POSITION": str(last)}
                result = subprocess.run(["/bin/bash", str(launcher)], env=env, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_seed_mode_executes_exact_global_tasks(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/run_mcd_repeated_correction_beam_full_cpucluster.sbatch"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            calls = root / "calls"
            fake_python = fake_bin / "python"
            fake_python.write_text(f'''#!/bin/bash
set -euo pipefail
args=("$@")
for ((i = 0; i < ${{#args[@]}}; i++)); do
  if [[ "${{args[i]}}" == --task-index ]]; then
    task="${{args[i + 1]}}"
  elif [[ "${{args[i]}}" == --task-count ]]; then
    count="${{args[i + 1]}}"
  fi
done
[[ -n "${{task:-}}" && "${{count:-}}" == 1599 ]]
while ! mkdir "{calls}.lock" 2>/dev/null; do sleep 0.001; done
echo "$task" >> "{calls}"
rmdir "{calls}.lock"
''')
            fake_python.chmod(0o755)
            fake_srun = fake_bin / "srun"
            fake_srun.write_text('''#!/bin/bash
set -euo pipefail
while [[ "$1" == -* ]]; do shift; done
[[ "$1" == /bin/bash && "$2" == -c ]]
script="$3"
shift 3
exec /bin/bash -c "$script" "$@"
''')
            fake_srun.chmod(0o755)
            base_env = {
                "BASE": str(launcher.parents[1]), "OUT": str(root / "out"),
                "SOURCE_SUBSETS_DIR": str(root / "subsets"), "MCD_MANIFEST_TREE": str(root / "manifest"),
                "MCD_STATE_ROOT": str(root / "state"), "MCD_GT_ROOT": str(root / "gt"),
                "CHECKPOINT_ROOT": str(root / "checkpoints"), "PYTHON": str(fake_python),
                "BEAM_CONFIG": str(launcher.parents[1] / "configs/evaluation/mcd_repeated_correction_beam_full_v1.json"),
                "FIRST_CLIP_POSITION": "0", "LAST_CLIP_POSITION": "532",
                "SLURM_JOB_ID": "test", "PATH": f"{fake_bin}:/usr/bin:/bin",
            }
            for seed in range(3):
                env = {**base_env, "RUN_SEED": str(seed)}
                result = subprocess.run(["/bin/bash", str(launcher)], env=env, text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                tasks = [int(value) for value in calls.read_text().splitlines()]
                self.assertEqual(sorted(tasks), list(range(seed, 1599, 3)))
                self.assertEqual(len(tasks), 533)
                calls.unlink()

    def test_invalid_ranges_fail(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/run_mcd_repeated_correction_beam_full_cpucluster.sbatch"
        with tempfile.TemporaryDirectory() as tmp:
            common = {
                "BASE": str(launcher.parents[1]), "OUT": str(Path(tmp) / "out"),
                "SOURCE_SUBSETS_DIR": str(Path(tmp) / "subsets"), "MCD_MANIFEST_TREE": str(Path(tmp) / "manifest"),
                "MCD_STATE_ROOT": str(Path(tmp) / "state"), "MCD_GT_ROOT": str(Path(tmp) / "gt"),
                "CHECKPOINT_ROOT": str(Path(tmp) / "checkpoints"), "PYTHON": "/bin/true",
                "SLURM_JOB_ID": "test", "PATH": "/usr/bin:/bin",
            }
            for first, last in ((0, 1598), (1, 799), (800, 1597)):
                env = {**common, "FIRST_TASK": str(first), "LAST_TASK": str(last)}
                result = subprocess.run(["/bin/bash", str(launcher)], env=env, text=True, capture_output=True)
                self.assertNotEqual(result.returncode, 0)

    def test_worker_failure_is_propagated(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/run_mcd_repeated_correction_beam_full_cpucluster.sbatch"
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir()
            counter = Path(tmp) / "counter"
            fake_srun = f'''#!/bin/bash
while ! mkdir "{counter}.lock" 2>/dev/null; do sleep 0.001; done
count=$(cat "{counter}" 2>/dev/null || echo 0)
count=$((count + 1))
echo "$count" > "{counter}"
rmdir "{counter}.lock"
if [[ "$count" == 3 ]]; then exit 7; fi
exit 0
'''
            (fake_bin / "srun").write_text(fake_srun)
            (fake_bin / "srun").chmod(0o755)
            env = {
                "BASE": str(launcher.parents[1]),
                "OUT": str(Path(tmp) / "out"),
                "SOURCE_SUBSETS_DIR": str(Path(tmp) / "subsets"),
                "MCD_MANIFEST_TREE": str(Path(tmp) / "manifest"),
                "MCD_STATE_ROOT": str(Path(tmp) / "state"),
                "MCD_GT_ROOT": str(Path(tmp) / "gt"),
                "CHECKPOINT_ROOT": str(Path(tmp) / "checkpoints"),
                "FIRST_TASK": "0",
                "LAST_TASK": "799",
                "PYTHON": "/bin/true",
                "SLURM_JOB_ID": "test",
                "PATH": f"{fake_bin}:/usr/bin:/bin",
            }
            result = subprocess.run(["/bin/bash", str(launcher)], env=env, text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)

    def test_nested_srun_exports_parent_environment(self):
        launcher = Path(__file__).resolve().parents[1] / "slurm/run_mcd_repeated_correction_beam_full_cpucluster.sbatch"
        with tempfile.TemporaryDirectory() as tmp:
            fake_bin = Path(tmp) / "bin"
            fake_bin.mkdir()
            marker = Path(tmp) / "marker"
            fake_srun = f'''#!/bin/bash
[[ "$1" == "--export=ALL" ]] || exit 91
shift
while [[ "$1" == -* ]]; do shift; done
[[ "$1" == "/bin/bash" && "$2" == "-c" ]] || exit 92
[[ -n "${{OUT:-}}" ]] || exit 93
printf '%s' "$OUT" > "{marker}"
exit 0
'''
            (fake_bin / "srun").write_text(fake_srun)
            (fake_bin / "srun").chmod(0o755)
            common = {
                "BASE": str(launcher.parents[1]), "OUT": str(Path(tmp) / "out"),
                "SOURCE_SUBSETS_DIR": str(Path(tmp) / "subsets"), "MCD_MANIFEST_TREE": str(Path(tmp) / "manifest"),
                "MCD_STATE_ROOT": str(Path(tmp) / "state"), "MCD_GT_ROOT": str(Path(tmp) / "gt"),
                "CHECKPOINT_ROOT": str(Path(tmp) / "checkpoints"), "FIRST_TASK": "0", "LAST_TASK": "799",
                "PYTHON": "/bin/true", "SLURM_JOB_ID": "test", "PATH": f"{fake_bin}:/usr/bin:/bin",
            }
            result = subprocess.run(["/bin/bash", str(launcher)], env={**common, "OUT": common["OUT"]}, text=True, capture_output=True)
            marker_text = marker.read_text()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(marker_text, common["OUT"])
