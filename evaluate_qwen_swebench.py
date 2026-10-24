#!/usr/bin/env python3
"""
Live Evaluator for Qwen models on Agentless (SWE-bench Lite).

Evaluates performance across all phases (matching the Agentless paper 2407.01489v2):
1. Phase 1: Fault Localization (File Top-1/Top-3/Top-5, Related Elements, Line Overlap)
2. Phase 2: Repair / Patch Generation (Diff validity, GT file/line edit coverage)
3. Phase 3: Patch Selection / Reranking metrics
4. High-resolution Timing & Token Accounting per step and task.

Outputs:
- Live terminal card per task
- results/evaluation/live_dashboard.md (Live updated Markdown dashboard)
- results/evaluation/live_metrics.json (Live updated machine-readable metrics)
- results/evaluation/live_summary.csv (Live updated spreadsheet)
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

# Ensure Agentless is in python path
WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
AGENTLESS_PATH = os.path.join(WORKSPACE_ROOT, "Agentless-main")
if AGENTLESS_PATH not in sys.path:
    sys.path.insert(0, AGENTLESS_PATH)

DEFAULT_GOLDENS_PATH = os.path.join(WORKSPACE_ROOT, "swe_bench_lite_goldens.json")
DEFAULT_RETRIEVAL_FILE = os.path.join(
    WORKSPACE_ROOT,
    "agentless_swebench_lite/retrievel_embedding/retrieve_locs.jsonl",
)


def parse_diff_patch(patch_str: str) -> Dict[str, Any]:
    """Parses a unified diff patch to extract files modified and line ranges."""
    if not patch_str or not isinstance(patch_str, str):
        return {"valid": False, "files": [], "lines_added": 0, "lines_removed": 0}

    files = []
    lines_added = 0
    lines_removed = 0
    is_valid = False

    for line in patch_str.split("\n"):
        if line.startswith("diff --git a/"):
            is_valid = True
            parts = line.split(" ")
            if len(parts) >= 4:
                file_path = parts[2][2:]  # strip a/
                if file_path not in files:
                    files.append(file_path)
        elif line.startswith("--- a/") or line.startswith("--- "):
            is_valid = True
            fn = line.split(" ", 1)[1].strip()
            if fn.startswith("a/"):
                fn = fn[2:]
            if fn and fn != "/dev/null" and fn not in files:
                files.append(fn)
        elif line.startswith("+++ b/") or line.startswith("+++ "):
            is_valid = True
            fn = line.split(" ", 1)[1].strip()
            if fn.startswith("b/"):
                fn = fn[2:]
            if fn and fn != "/dev/null" and fn not in files:
                files.append(fn)
        elif line.startswith("+") and not line.startswith("+++"):
            lines_added += 1
        elif line.startswith("-") and not line.startswith("---"):
            lines_removed += 1

    return {
        "valid": is_valid and (lines_added > 0 or lines_removed > 0),
        "files": files,
        "lines_added": lines_added,
        "lines_removed": lines_removed,
    }


def evaluate_fl_file_match(found_files: List[str], gt_files: List[str]) -> Dict[str, Any]:
    """Evaluates file-level localization against ground truth files."""
    if not gt_files:
        return {"top_1": False, "top_3": False, "top_5": False, "all_hit": False, "hit_rank": None}

    hit_rank = None
    for idx, f in enumerate(found_files):
        if f in gt_files:
            hit_rank = idx + 1
            break

    top_1 = hit_rank == 1 if hit_rank else False
    top_3 = hit_rank is not None and hit_rank <= 3
    top_5 = hit_rank is not None and hit_rank <= 5
    all_hit = all(gt in found_files for gt in gt_files)

    return {
        "top_1": top_1,
        "top_3": top_3,
        "top_5": top_5,
        "all_hit": all_hit,
        "hit_rank": hit_rank,
    }


def evaluate_fl_line_overlap(found_edit_locs: Any, gt_lines: Dict[str, List[Dict[str, int]]]) -> bool:
    """Checks if fine-grained edit locations overlap with ground truth modified lines."""
    if not found_edit_locs or not gt_lines:
        return False

    dict_list = []
    if isinstance(found_edit_locs, list):
        for item in found_edit_locs:
            if isinstance(item, dict):
                dict_list.append(item)
    elif isinstance(found_edit_locs, dict):
        dict_list.append(found_edit_locs)

    for loc_dict in dict_list:
        for file_path, loc_entries in loc_dict.items():
            if file_path not in gt_lines:
                continue
            gt_intervals = gt_lines[file_path]
            if isinstance(loc_entries, str):
                loc_entries = [loc_entries]
            for entry in loc_entries:
                # Extract line numbers from entries like "line: 307" or "line: 304-310"
                line_matches = re.findall(r"line:\s*(\d+)(?:-(\d+))?", str(entry))
                for start_str, end_str in line_matches:
                    p_start = int(start_str)
                    p_end = int(end_str) if end_str else p_start
                    for gt in gt_intervals:
                        g_start = gt["start"]
                        g_end = gt["end"]
                        # Check overlap [p_start, p_end] with [g_start, g_end]
                        if max(p_start, g_start) <= min(p_end, g_end):
                            return True
    return False


def run_command_timed(cmd_list: List[str], cwd: str, env: Dict[str, str]) -> Dict[str, Any]:
    """Runs a command, captures wall-clock time, stdout, stderr, and exit code."""
    t0 = time.perf_counter()
    res = subprocess.run(cmd_list, cwd=cwd, env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0
    return {
        "elapsed_s": round(elapsed, 2),
        "returncode": res.returncode,
        "stdout": res.stdout,
        "stderr": res.stderr,
    }


class LiveEvaluator:
    def __init__(
        self,
        model_name: str = "qwen3.5-4b-super-coder",
        base_url: str = "http://127.0.0.1:1234/v1",
        backend: str = "qwen",
        output_dir: str = "results/eval_qwen_swebench",
        goldens_file: str = DEFAULT_GOLDENS_PATH,
        retrieval_loc_file: str = DEFAULT_RETRIEVAL_FILE,
        python_bin: str = sys.executable,
        max_tokens: int = 1500,
    ):
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        if not self.base_url.endswith("/v1"):
            self.base_url = f"{self.base_url}/v1"
        self.backend = backend
        self.output_dir = os.path.abspath(output_dir)
        self.goldens_file = goldens_file
        self.retrieval_loc_file = retrieval_loc_file
        self.python_bin = python_bin
        self.max_tokens = max_tokens

        os.makedirs(self.output_dir, exist_ok=True)
        self.dashboard_md = os.path.join(self.output_dir, "live_dashboard.md")
        self.metrics_json = os.path.join(self.output_dir, "live_metrics.json")
        self.summary_csv = os.path.join(self.output_dir, "live_summary.csv")

        self.goldens = self.load_goldens()
        self.results: List[Dict[str, Any]] = self.load_existing_results()
        self.start_time = datetime.now().isoformat()

    def load_goldens(self) -> Dict[str, Any]:
        if os.path.exists(self.goldens_file):
            with open(self.goldens_file, "r") as f:
                return json.load(f)
        return {}

    def load_existing_results(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.metrics_json):
            try:
                with open(self.metrics_json, "r") as f:
                    data = json.load(f)
                    return data.get("tasks", [])
            except Exception:
                return []
        return []

    def get_completed_instance_ids(self) -> set:
        return {r["instance_id"] for r in self.results}

    def update_live_reports(self):
        """Atomically updates the Markdown dashboard, JSON metrics, and CSV summary."""
        total = len(self.results)
        if total == 0:
            return

        # Compute aggregate KPIs
        file_top1 = sum(1 for r in self.results if r.get("fl_file_top1"))
        file_top3 = sum(1 for r in self.results if r.get("fl_file_top3"))
        file_top5 = sum(1 for r in self.results if r.get("fl_file_top5"))
        line_hits = sum(1 for r in self.results if r.get("fl_line_hit"))
        valid_patches = sum(1 for r in self.results if r.get("repair_valid_diff"))
        touch_gt_patches = sum(1 for r in self.results if r.get("repair_touches_gt"))

        total_time_s = sum(r.get("total_time_s", 0) for r in self.results)
        avg_time_s = round(total_time_s / total, 1) if total > 0 else 0
        total_prompt_tok = sum(r.get("total_prompt_tokens", 0) for r in self.results)
        total_comp_tok = sum(r.get("total_completion_tokens", 0) for r in self.results)
        total_tok = total_prompt_tok + total_comp_tok

        metrics_data = {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "backend": self.backend,
            "last_updated": datetime.now().isoformat(),
            "summary": {
                "tasks_evaluated": total,
                "file_loc_top1_pct": round(file_top1 / total * 100, 1),
                "file_loc_top3_pct": round(file_top3 / total * 100, 1),
                "file_loc_top5_pct": round(file_top5 / total * 100, 1),
                "line_loc_overlap_pct": round(line_hits / total * 100, 1),
                "valid_diff_rate_pct": round(valid_patches / total * 100, 1),
                "touches_gt_patch_pct": round(touch_gt_patches / total * 100, 1),
                "avg_task_latency_s": avg_time_s,
                "total_prompt_tokens": total_prompt_tok,
                "total_completion_tokens": total_comp_tok,
                "total_tokens": total_tok,
            },
            "tasks": self.results,
        }

        # 1. Write JSON
        tmp_json = f"{self.metrics_json}.tmp"
        with open(tmp_json, "w") as f:
            json.dump(metrics_data, f, indent=2)
        os.replace(tmp_json, self.metrics_json)

        # 2. Write CSV
        tmp_csv = f"{self.summary_csv}.tmp"
        fieldnames = [
            "instance_id", "status", "fl_file_top1", "fl_file_top3", "fl_file_top5",
            "fl_line_hit", "repair_valid_diff", "repair_touches_gt",
            "diff_lines_added", "diff_lines_removed",
            "t_file_s", "t_combine_s", "t_related_s", "t_line_s", "t_repair_s", "total_time_s",
            "prompt_tokens", "completion_tokens", "total_tokens", "timestamp"
        ]
        with open(tmp_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self.results:
                writer.writerow({
                    "instance_id": r.get("instance_id"),
                    "status": r.get("status"),
                    "fl_file_top1": r.get("fl_file_top1"),
                    "fl_file_top3": r.get("fl_file_top3"),
                    "fl_file_top5": r.get("fl_file_top5"),
                    "fl_line_hit": r.get("fl_line_hit"),
                    "repair_valid_diff": r.get("repair_valid_diff"),
                    "repair_touches_gt": r.get("repair_touches_gt"),
                    "diff_lines_added": r.get("diff_lines_added", 0),
                    "diff_lines_removed": r.get("diff_lines_removed", 0),
                    "t_file_s": r.get("timings", {}).get("file_level", 0),
                    "t_combine_s": r.get("timings", {}).get("combine", 0),
                    "t_related_s": r.get("timings", {}).get("related", 0),
                    "t_line_s": r.get("timings", {}).get("line_level", 0),
                    "t_repair_s": r.get("timings", {}).get("repair", 0),
                    "total_time_s": r.get("total_time_s", 0),
                    "prompt_tokens": r.get("total_prompt_tokens", 0),
                    "completion_tokens": r.get("total_completion_tokens", 0),
                    "total_tokens": r.get("total_tokens", 0),
                    "timestamp": r.get("timestamp"),
                })
        os.replace(tmp_csv, self.summary_csv)

        # 3. Write Markdown Dashboard
        tmp_md = f"{self.dashboard_md}.tmp"
        with open(tmp_md, "w") as f:
            f.write(f"# 📊 Agentless Live Evaluation Dashboard\n\n")
            f.write(f"**Model:** `{self.model_name}` | **Backend:** `{self.backend}` | **Base URL:** `{self.base_url}`\n")
            f.write(f"**Last Updated:** `{metrics_data['last_updated']}` | **Tasks Processed:** `{total}`\n\n")

            f.write("## 🎯 Performance Metrics Across All Phases (Paper Align)\n\n")
            f.write("| Metric | Paper Phase | Performance | Count |\n")
            f.write("| :--- | :--- | :--- | :--- |\n")
            f.write(f"| **File Localization (Top-1)** | Phase 1 (File FL) | **{metrics_data['summary']['file_loc_top1_pct']}%** | {file_top1}/{total} |\n")
            f.write(f"| **File Localization (Top-3)** | Phase 1 (File FL) | **{metrics_data['summary']['file_loc_top3_pct']}%** | {file_top3}/{total} |\n")
            f.write(f"| **File Localization (Top-5)** | Phase 1 (File FL) | **{metrics_data['summary']['file_loc_top5_pct']}%** | {file_top5}/{total} |\n")
            f.write(f"| **Edit Line Overlap** | Phase 1 (Line FL) | **{metrics_data['summary']['line_loc_overlap_pct']}%** | {line_hits}/{total} |\n")
            f.write(f"| **Valid Git Diff Rate** | Phase 2 (Repair) | **{metrics_data['summary']['valid_diff_rate_pct']}%** | {valid_patches}/{total} |\n")
            f.write(f"| **Touches GT Modified File** | Phase 2 (Repair) | **{metrics_data['summary']['touches_gt_patch_pct']}%** | {touch_gt_patches}/{total} |\n")
            f.write(f"| **Average Latency per Task** | Efficiency | **{avg_time_s}s** | Total: {round(total_time_s/60, 1)}m |\n")
            f.write(f"| **Total Tokens Consumed** | Efficiency | **{total_tok:,}** | P: {total_prompt_tok:,} / C: {total_comp_tok:,} |\n\n")

            f.write("## 📋 Task-by-Task Real-Time Results\n\n")
            f.write("| # | Task ID | File Top-1 | File Top-3 | Line Match | Diff Valid | Touches GT | Tokens | Latency | Status |\n")
            f.write("| :-: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for idx, r in enumerate(reversed(self.results[-50:])):  # Show latest 50
                f_top1 = "✓" if r.get("fl_file_top1") else "✗"
                f_top3 = "✓" if r.get("fl_file_top3") else "✗"
                l_hit = "✓" if r.get("fl_line_hit") else "✗"
                d_val = "✓" if r.get("repair_valid_diff") else "✗"
                t_gt = "✓" if r.get("repair_touches_gt") else "✗"
                tok = f"{r.get('total_tokens', 0):,}"
                lat = f"{r.get('total_time_s', 0)}s"
                status = f"**{r.get('status', 'DONE')}**"
                f.write(f"| {total - idx} | `{r['instance_id']}` | {f_top1} | {f_top3} | {l_hit} | {d_val} | {t_gt} | {tok} | {lat} | {status} |\n")

            if total > 50:
                f.write(f"\n*(Showing latest 50 tasks. See `live_summary.csv` for all {total} tasks)*\n")

        os.replace(tmp_md, self.dashboard_md)

    def print_task_card(self, task_result: Dict[str, Any]):
        """Prints an informative console summary card after each task completes."""
        inst = task_result["instance_id"]
        status = task_result["status"]
        f1 = "✓" if task_result.get("fl_file_top1") else "✗"
        f3 = "✓" if task_result.get("fl_file_top3") else "✗"
        line_h = "✓" if task_result.get("fl_line_hit") else "✗"
        diff_v = "✓" if task_result.get("repair_valid_diff") else "✗"
        gt_t = "✓" if task_result.get("repair_touches_gt") else "✗"
        tok = task_result.get("total_tokens", 0)
        tot_s = task_result.get("total_time_s", 0)
        timings = task_result.get("timings", {})

        print("\n" + "=" * 70)
        print(f" 🎯 TASK COMPLETED: {inst}  [{status}]")
        print("=" * 70)
        print(f"  Phase 1 (FL)     : File Top-1 [{f1}] | File Top-3 [{f3}] | Line Overlap [{line_h}]")
        print(f"  Phase 2 (Repair) : Valid Diff [{diff_v}] | Touches GT File [{gt_t}] | (+{task_result.get('diff_lines_added', 0)} / -{task_result.get('diff_lines_removed', 0)} lines)")
        print(f"  Execution Time   : {tot_s}s  (File: {timings.get('file_level', 0)}s, Combine: {timings.get('combine', 0)}s, Elements: {timings.get('related', 0)}s, LineLoc: {timings.get('line_level', 0)}s, Repair: {timings.get('repair', 0)}s)")
        print(f"  Token Usage      : {tok:,} total (Prompt: {task_result.get('total_prompt_tokens', 0):,}, Generated: {task_result.get('total_completion_tokens', 0):,})")
        print(f"  Live Artifacts   : {self.dashboard_md}")
        print("=" * 70 + "\n")

    def run_task(self, target_id: str, force: bool = False, reuse_fl: bool = False) -> Dict[str, Any]:
        """Executes all Agentless phases for a single SWE-bench instance."""
        task_out_base = os.path.join(self.output_dir, target_id)
        if force and not reuse_fl and os.path.exists(task_out_base):
            import shutil
            shutil.rmtree(task_out_base)
        os.makedirs(task_out_base, exist_ok=True)

        if reuse_fl:
            stage6_old = os.path.join(task_out_base, "repair_sample_1")
            if os.path.exists(stage6_old):
                import shutil
                shutil.rmtree(stage6_old)

        gt = self.goldens.get(target_id, {})
        gt_files = gt.get("gt_files", [])
        gt_lines = gt.get("gt_lines", {})

        env = os.environ.copy()
        env["PYTHONPATH"] = f"{env.get('PYTHONPATH', '')}:{AGENTLESS_PATH}"
        env["OPENAI_BASE_URL"] = self.base_url
        env["OPENAI_API_KEY"] = "EMPTY"

        timings = {}
        total_prompt_tokens = 0
        total_completion_tokens = 0
        t_task_start = time.perf_counter()

        # -------------------------------------------------------------
        # Stage 1: File-level Localization
        # -------------------------------------------------------------
        stage1_out = os.path.join(task_out_base, "file_level")
        cmd_s1 = [
            self.python_bin,
            os.path.join(AGENTLESS_PATH, "agentless/fl/localize.py"),
            "--file_level",
            "--target_id", target_id,
            "--output_folder", stage1_out,
            "--model", self.model_name,
            "--backend", self.backend,
            "--base_url", self.base_url,
            "--max_tokens", str(self.max_tokens),
            "--num_threads", "1",
            "--skip_existing",
        ]
        res1 = run_command_timed(cmd_s1, cwd=WORKSPACE_ROOT, env=env)
        timings["file_level"] = res1["elapsed_s"]
        if res1["returncode"] != 0 and res1["stderr"]:
            print(f"  [Stage 1 Stderr]: {res1['stderr'].strip()[:300]}")

        # Parse Stage 1 outputs
        found_files = []
        loc_file_s1 = os.path.join(stage1_out, "loc_outputs.jsonl")
        if os.path.exists(loc_file_s1):
            with open(loc_file_s1) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("instance_id") == target_id:
                            found_files = entry.get("found_files", [])
                            traj = entry.get("file_traj", {}).get("usage", {})
                            total_prompt_tokens += traj.get("prompt_tokens", 0)
                            total_completion_tokens += traj.get("completion_tokens", 0)
                    except Exception:
                        pass

        # -------------------------------------------------------------
        # Stage 2: Combine with Precomputed Retrieval
        # -------------------------------------------------------------
        stage2_out = os.path.join(task_out_base, "file_level_combined")
        combined_loc_file = os.path.join(stage2_out, "combined_locs.jsonl")
        retrieval_file = self.retrieval_loc_file

        if os.path.exists(combined_loc_file):
            try:
                os.remove(combined_loc_file)
            except OSError:
                pass

        if os.path.exists(retrieval_file) and os.path.exists(loc_file_s1):
            cmd_s2 = [
                self.python_bin,
                os.path.join(AGENTLESS_PATH, "agentless/fl/combine.py"),
                "--retrieval_loc_file", retrieval_file,
                "--model_loc_file", loc_file_s1,
                "--top_n", "3",
                "--output_folder", stage2_out,
            ]
            res2 = run_command_timed(cmd_s2, cwd=WORKSPACE_ROOT, env=env)
            timings["combine"] = res2["elapsed_s"]
            if res2["returncode"] != 0 and res2["stderr"]:
                print(f"  [Stage 2 Stderr]: {res2['stderr'].strip()[:300]}")
        else:
            timings["combine"] = 0.0

        if not os.path.exists(combined_loc_file):
            combined_loc_file = loc_file_s1

        # -------------------------------------------------------------
        # Stage 3: Related Elements Localization
        # -------------------------------------------------------------
        stage3_out = os.path.join(task_out_base, "related_elements")
        cmd_s3 = [
            self.python_bin,
            os.path.join(AGENTLESS_PATH, "agentless/fl/localize.py"),
            "--related_level",
            "--target_id", target_id,
            "--output_folder", stage3_out,
            "--top_n", "3",
            "--compress_assign",
            "--compress",
            "--start_file", combined_loc_file,
            "--model", self.model_name,
            "--backend", self.backend,
            "--base_url", self.base_url,
            "--max_tokens", str(self.max_tokens),
            "--num_threads", "1",
            "--skip_existing",
        ]
        res3 = run_command_timed(cmd_s3, cwd=WORKSPACE_ROOT, env=env)
        timings["related"] = res3["elapsed_s"]
        if res3["returncode"] != 0 and res3["stderr"]:
            print(f"  [Stage 3 Stderr]: {res3['stderr'].strip()[:300]}")

        loc_file_s3 = os.path.join(stage3_out, "loc_outputs.jsonl")
        if os.path.exists(loc_file_s3):
            with open(loc_file_s3) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("instance_id") == target_id:
                            traj = entry.get("related_loc_traj", {}).get("usage", {})
                            total_prompt_tokens += traj.get("prompt_tokens", 0)
                            total_completion_tokens += traj.get("completion_tokens", 0)
                    except Exception:
                        pass
        else:
            loc_file_s3 = combined_loc_file

        # -------------------------------------------------------------
        # Stage 4: Fine-grain Edit Location Localization
        # -------------------------------------------------------------
        stage4_out = os.path.join(task_out_base, "edit_locations")
        cmd_s4 = [
            self.python_bin,
            os.path.join(AGENTLESS_PATH, "agentless/fl/localize.py"),
            "--fine_grain_line_level",
            "--target_id", target_id,
            "--output_folder", stage4_out,
            "--top_n", "3",
            "--compress",
            "--temperature", "0.7",
            "--num_samples", "2",
            "--start_file", loc_file_s3,
            "--model", self.model_name,
            "--backend", self.backend,
            "--base_url", self.base_url,
            "--max_tokens", str(self.max_tokens),
            "--num_threads", "1",
            "--skip_existing",
        ]
        res4 = run_command_timed(cmd_s4, cwd=WORKSPACE_ROOT, env=env)
        timings["line_level"] = res4["elapsed_s"]
        if res4["returncode"] != 0 and res4["stderr"]:
            print(f"  [Stage 4 Stderr]: {res4['stderr'].strip()[:300]}")

        found_edit_locs = {}
        loc_file_s4 = os.path.join(stage4_out, "loc_outputs.jsonl")
        if os.path.exists(loc_file_s4):
            with open(loc_file_s4) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("instance_id") == target_id:
                            found_edit_locs = entry.get("found_edit_locs", {})
                            traj = entry.get("edit_loc_traj", {}).get("usage", {})
                            total_prompt_tokens += traj.get("prompt_tokens", 0)
                            total_completion_tokens += traj.get("completion_tokens", 0)
                    except Exception:
                        pass
        else:
            loc_file_s4 = loc_file_s3

        # -------------------------------------------------------------
        # Stage 5: Merge Edit Locations
        # -------------------------------------------------------------
        stage5_out = os.path.join(task_out_base, "edit_location_individual")
        cmd_s5 = [
            self.python_bin,
            os.path.join(AGENTLESS_PATH, "agentless/fl/localize.py"),
            "--merge",
            "--output_folder", stage5_out,
            "--top_n", "3",
            "--num_samples", "2",
            "--start_file", loc_file_s4,
        ]
        res5 = run_command_timed(cmd_s5, cwd=WORKSPACE_ROOT, env=env)
        timings["merge"] = res5["elapsed_s"]
        if res5["returncode"] != 0 and res5["stderr"]:
            print(f"  [Stage 5 Stderr]: {res5['stderr'].strip()[:300]}")

        # -------------------------------------------------------------
        # Stage 6: Repair / Patch Generation
        # -------------------------------------------------------------
        merged_loc_file = os.path.join(stage5_out, "loc_merged_0-0_outputs.jsonl")
        if not os.path.exists(merged_loc_file):
            merged_loc_file = loc_file_s4

        stage6_out = os.path.join(task_out_base, "repair_sample_1")
        cmd_s6 = [
            self.python_bin,
            os.path.join(AGENTLESS_PATH, "agentless/repair/repair.py"),
            "--loc_file", merged_loc_file,
            "--output_folder", stage6_out,
            "--target_id", target_id,
            "--loc_interval",
            "--top_n", "3",
            "--context_window", "10",
            "--max_samples", "2",
            "--cot",
            "--diff_format",
            "--gen_and_process",
            "--model", self.model_name,
            "--backend", self.backend,
            "--base_url", self.base_url,
            "--max_tokens", str(self.max_tokens),
            "--num_threads", "1",
        ]
        res6 = run_command_timed(cmd_s6, cwd=WORKSPACE_ROOT, env=env)
        timings["repair"] = res6["elapsed_s"]
        if res6["returncode"] != 0 and res6["stderr"]:
            print(f"  [Stage 6 Stderr]: {res6['stderr'].strip()[:300]}")

        # Parse generated patch
        generated_patch = ""
        patch_file = os.path.join(stage6_out, "output_0_processed.jsonl")
        if os.path.exists(patch_file):
            with open(patch_file) as f:
                for line in f:
                    try:
                        entry = json.loads(line)
                        if entry.get("instance_id") == target_id:
                            generated_patch = entry.get("model_patch", "")
                            break
                    except Exception:
                        pass

        total_task_time = round(time.perf_counter() - t_task_start, 2)

        # Evaluate against Ground Truth
        fl_eval = evaluate_fl_file_match(found_files, gt_files)
        line_overlap = evaluate_fl_line_overlap(found_edit_locs, gt_lines)
        diff_info = parse_diff_patch(generated_patch)

        touches_gt = any(f in gt_files for f in diff_info["files"]) if gt_files else False
        status = "SUCCESS" if (fl_eval["top_3"] and diff_info["valid"]) else "PARTIAL" if fl_eval["top_5"] else "FAILED"

        task_record = {
            "instance_id": target_id,
            "status": status,
            "fl_file_top1": fl_eval["top_1"],
            "fl_file_top3": fl_eval["top_3"],
            "fl_file_top5": fl_eval["top_5"],
            "fl_file_rank": fl_eval["hit_rank"],
            "fl_line_hit": line_overlap,
            "repair_valid_diff": diff_info["valid"],
            "repair_touches_gt": touches_gt,
            "diff_lines_added": diff_info["lines_added"],
            "diff_lines_removed": diff_info["lines_removed"],
            "timings": timings,
            "total_time_s": total_task_time,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_prompt_tokens + total_completion_tokens,
            "timestamp": datetime.now().isoformat(),
        }

        self.results.append(task_record)
        self.update_live_reports()
        self.print_task_card(task_record)
        return task_record

    def run_benchmark(self, target_ids: Optional[List[str]] = None, limit: Optional[int] = None, force: bool = False, reuse_fl: bool = False):
        """Runs evaluation over a list of tasks with live streaming results."""
        if target_ids is None or len(target_ids) == 0:
            target_ids = list(self.goldens.keys())

        if limit:
            target_ids = target_ids[:limit]

        if force or reuse_fl:
            # Filter out existing entries for the specified targets
            self.results = [r for r in self.results if r["instance_id"] not in target_ids]
            completed = set()
            remaining = target_ids
        else:
            completed = self.get_completed_instance_ids()
            remaining = [t for t in target_ids if t not in completed]

        print("\n" + "=" * 70)
        print(" 🚀 AGENTLESS LIVE EVALUATION ENGINE")
        print("=" * 70)
        print(f" Model           : {self.model_name}")
        print(f" Backend         : {self.backend}")
        print(f" Server Base URL : {self.base_url}")
        print(f" Output Folder   : {self.output_dir}")
        print(f" Total Tasks     : {len(target_ids)} ({len(completed)} already completed, {len(remaining)} pending)")
        print(f" Live Dashboard  : {self.dashboard_md}")
        print("=" * 70 + "\n")

        for idx, target_id in enumerate(remaining, 1):
            print(f"\n[{idx}/{len(remaining)}] Starting evaluation for {target_id}...")
            try:
                self.run_task(target_id, force=force, reuse_fl=reuse_fl)
            except Exception as e:
                print(f"[!] Error evaluating {target_id}: {e}")
                err_record = {
                    "instance_id": target_id,
                    "status": "ERROR",
                    "error": str(e),
                    "total_time_s": 0,
                    "timestamp": datetime.now().isoformat(),
                }
                self.results.append(err_record)
                self.update_live_reports()

        print("\n" + "=" * 70)
        print(" 🎉 EVALUATION COMPLETED!")
        print(f" Dashboard Summary: {self.dashboard_md}")
        print(f" Machine Metrics  : {self.metrics_json}")
        print(f" CSV Export       : {self.summary_csv}")
        print("=" * 70 + "\n")


def print_live_summary(metrics_path: str):
    """Prints a terminal summary of the live metrics file."""
    if not os.path.exists(metrics_path):
        print(f"No metrics found at {metrics_path}. Please run an evaluation first.")
        return

    with open(metrics_path) as f:
        data = json.load(f)

    summ = data.get("summary", {})
    tasks = data.get("tasks", [])
    print("\n" + "=" * 70)
    print(f" 📊 LIVE METRICS SNAPSHOT: {data.get('model_name')}")
    print("=" * 70)
    print(f" Last Updated        : {data.get('last_updated')}")
    print(f" Tasks Evaluated     : {summ.get('tasks_evaluated')}")
    print(f" File Loc Top-1 Rate : {summ.get('file_loc_top1_pct')}%")
    print(f" File Loc Top-3 Rate : {summ.get('file_loc_top3_pct')}%")
    print(f" File Loc Top-5 Rate : {summ.get('file_loc_top5_pct')}%")
    print(f" Line Loc Overlap    : {summ.get('line_loc_overlap_pct')}%")
    print(f" Valid Diff Rate     : {summ.get('valid_diff_rate_pct')}%")
    print(f" Touches GT Patch    : {summ.get('touches_gt_patch_pct')}%")
    print(f" Avg Latency / Task  : {summ.get('avg_task_latency_s')}s")
    print(f" Total Tokens        : {summ.get('total_tokens', 0):,}")
    print("=" * 70)
    print(" Recent Tasks:")
    for t in tasks[-10:]:
        print(f"  * {t['instance_id']:<28} [{t['status']:<7}] FL_Top3: {str(t.get('fl_file_top3')):<5} DiffValid: {str(t.get('repair_valid_diff')):<5} Time: {t.get('total_time_s', 0)}s")
    print("=" * 70 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Live Evaluator for Agentless on SWE-bench Lite")
    parser.add_argument("--model_name", type=str, default=os.getenv("MODEL_NAME", "qwen3.5-4b-super-coder"))
    parser.add_argument("--base_url", type=str, default=os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:1234/v1"))
    parser.add_argument("--backend", type=str, default="qwen")
    parser.add_argument("--target_id", type=str, help="Target single bug (e.g. django__django-10914)")
    parser.add_argument("--target_ids", type=str, help="Comma-separated bug IDs to evaluate")
    parser.add_argument("--filter", type=str, help="Filter tasks by repo or substring (e.g. 'django', 'sympy', 'pytest')")
    parser.add_argument("--limit", type=int, help="Limit to first N tasks")
    parser.add_argument("--max_tokens", type=int, default=1500, help="Max tokens for LLM generation")
    parser.add_argument("--output_dir", type=str, default="results/eval_qwen_swebench")
    parser.add_argument("--force", action="store_true", help="Re-evaluate tasks even if previously completed")
    parser.add_argument("--reuse_fl", action="store_true", help="Reuse existing Phase 1 Fault Localization outputs and re-run Phase 2 Repair only")
    parser.add_argument("--report", action="store_true", help="Print live dashboard report from existing metrics and exit")

    args = parser.parse_args()

    metrics_file = os.path.join(args.output_dir, "live_metrics.json")
    if args.report:
        print_live_summary(metrics_file)
        return

    evaluator = LiveEvaluator(
        model_name=args.model_name,
        base_url=args.base_url,
        backend=args.backend,
        output_dir=args.output_dir,
        max_tokens=args.max_tokens,
    )

    targets = None
    if args.target_id:
        targets = [args.target_id]
    elif args.target_ids:
        targets = [t.strip() for t in args.target_ids.split(",") if t.strip()]
    elif args.filter:
        targets = [t for t in evaluator.goldens.keys() if args.filter.lower() in t.lower()]
        print(f"Filtered {len(targets)} tasks matching '{args.filter}'")

    evaluator.run_benchmark(target_ids=targets, limit=args.limit, force=args.force, reuse_fl=args.reuse_fl)


if __name__ == "__main__":
    main()
