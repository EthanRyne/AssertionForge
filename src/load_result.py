%%writefile /content/AssertionForge/src/load_result.py

# load_result.py
from __future__ import annotations
import os
import re
import json
from pathlib import Path
from typing import Dict, List

# Use repo logger if available
try:
    from saver import saver
    log = saver.log_info
except Exception:
    log = print


def load_pdf_stats(load_dir: str) -> dict:
    """
    Load PDF statistics from a previous run.
    Looks for <load_dir>/pdf_stats.json. If not found, returns a safe default.
    """
    load_dir = Path(load_dir)
    stats_path = load_dir / "pdf_stats.json"

    if stats_path.exists():
        try:
            with open(stats_path, "r", encoding="utf-8") as f:
                stats = json.load(f)
            # Ensure the expected keys exist; fill missing with 0
            for k in ["num_pages", "num_tokens", "file_size", "num_files"]:
                stats.setdefault(k, 0)
            log(f"Loaded PDF stats from {stats_path}")
            return stats
        except Exception as e:
            log(f"Failed to read {stats_path}: {e}")

    # Fallback: safe defaults if no persisted stats are available
    log("pdf_stats.json not found; using default zeros.")
    return {
        "num_pages": 0,
        "num_tokens": 0,
        "file_size": 0,
        "num_files": 0,
    }


def load_nl_plans(load_dir: str) -> Dict[str, List[str]]:
    """
    Load NL plans from a previous run.

    Preferred:
      <load_dir>/nl_plans.json   -> { signal: [plan, ...], ... }

    Fallback:
      <load_dir>/nl_plans.txt    -> sections like:
          Signal PWDATA:
          Plan 1: that when ...
          Plan 2: that ...

    Returns:
      Dict[str, List[str]]
    """
    load_dir = Path(load_dir)
    # Preferred JSON
    json_path = load_dir / "nl_plans.json"
    if json_path.exists():
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # normalize values to list[str]
            norm = {str(k): [str(p) for p in v] for k, v in data.items()}
            log(f"Loaded NL plans from {json_path}")
            return norm
        except Exception as e:
            log(f"Failed to parse {json_path}: {e}")

    # Fallback TXT (format written in gen_plan())
    txt_path = load_dir / "nl_plans.txt"
    plans: Dict[str, List[str]] = {}
    if txt_path.exists():
        try:
            current_signal = None
            with open(txt_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line:
                        continue
                    # Section header: "Signal <name>:"
                    m_sig = re.match(r"^Signal\s+(.+?):\s*$", line)
                    if m_sig:
                        current_signal = m_sig.group(1).strip()
                        plans.setdefault(current_signal, [])
                        continue
                    # Plan line: "Plan N: <text>"
                    m_plan = re.match(r"^Plan\s+\d+\s*:\s*(.+)$", line)
                    if m_plan and current_signal:
                        plans[current_signal].append(m_plan.group(1).strip())
                        continue
                    # Continuation line for previous plan (rare, but be tolerant)
                    if current_signal and plans.get(current_signal):
                        plans[current_signal][-1] = (plans[current_signal][-1] + " " + line).strip()

            log(f"Loaded NL plans from {txt_path}")
            return plans
        except Exception as e:
            log(f"Failed to parse {txt_path}: {e}")

    log("No NL plans found; returning empty dict.")
    return {}


def load_svas(load_dir: str) -> List[str]:
    """
    Load generated SVAs from <load_dir>/tbs/*.sva.

    We try to return just the SVA property body (between 'property ...;' and 'endproperty').
    If that extraction fails, we return the whole file content to keep downstream steps working.
    """
    load_dir = Path(load_dir)
    tbs_dir = load_dir / "tbs"
    if not tbs_dir.exists():
        log(f"{tbs_dir} does not exist; returning empty SVA list.")
        return []

    # Sort files deterministically, prefer property_goldmine_<i>.sva ordering
    files = sorted(tbs_dir.glob("property_goldmine_*.sva"), key=_filename_index_key)
    if not files:
        # Fallback: any .sva files
        files = sorted(tbs_dir.glob("*.sva"))

    svas: List[str] = []
    prop_re = re.compile(r"property\s+\w+\s*;\s*(.*?)\s*endproperty", re.DOTALL | re.IGNORECASE)

    for p in files:
        try:
            with open(p, "r", encoding="utf-8") as f:
                text = f.read()
            m = prop_re.search(text)
            if m:
                body = m.group(1).strip()
                svas.append(body)
            else:
                # Couldn’t isolate the body; keep the full text to preserve information
                svas.append(text.strip())
        except Exception as e:
            log(f"Failed to read {p}: {e}")

    log(f"Loaded {len(svas)} SVAs from {tbs_dir}")
    return svas


def load_jasper_reports(load_dir: str) -> List[str]:
    """
    Return paths to JasperGold report files in <load_dir>/jasper_reports/*.txt
    ordered by their numeric suffix if present (jasper_report_<i>.txt).
    """
    load_dir = Path(load_dir)
    rpt_dir = load_dir / "jasper_reports"
    if not rpt_dir.exists():
        log(f"{rpt_dir} does not exist; returning empty report list.")
        return []

    # Prefer jasper_report_<i>.txt sorted by <i>
    reports = sorted(rpt_dir.glob("jasper_report_*.txt"), key=_filename_index_key)
    if not reports:
        reports = sorted(rpt_dir.glob("*.txt"))

    paths = [str(p) for p in reports]
    log(f"Found {len(paths)} Jasper reports in {rpt_dir}")
    return paths


# -------------------------
# Helpers
# -------------------------

_idx_re = re.compile(r"_(\d+)\D*$")

def _filename_index_key(path: Path):
    """
    Extract trailing integer to sort files like foo_0.sva, foo_1.sva, ...
    Falls back to filename sort.
    """
    m = _idx_re.search(path.stem)
    if m:
        try:
            return int(m.group(1))
        except Exception:
            pass
    return path.name
