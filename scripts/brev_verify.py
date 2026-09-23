"""Run the reproducibility checks and save non-secret NVIDIA Brev evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone


REPO_ROOT = Path(__file__).resolve().parents[1]


def run_command(name: str, command: list[str], output_dir: Path) -> dict:
    started = time.monotonic()
    result = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    duration = time.monotonic() - started
    log_path = output_dir / f"{name}.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    log_sha256 = hashlib.sha256(result.stdout.encode("utf-8")).hexdigest()
    print(f"[{name}] exit={result.returncode} duration={duration:.2f}s log={log_path}")
    return {
        "command": command,
        "exit_code": result.returncode,
        "duration_seconds": round(duration, 3),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "log_sha256": log_sha256,
    }


def capture(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=REPO_ROOT,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_versions() -> dict[str, str]:
    import numpy
    import pandas

    return {"numpy": numpy.__version__, "pandas": pandas.__version__}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output-dir", default=".brev-evidence")
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    if args.runs <= 0:
        parser.error("--runs must be positive")

    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    git_status_before = capture(["git", "status", "--porcelain"]) or ""

    python = sys.executable
    steps = {
        "unittest": run_command(
            "unittest",
            [python, "-m", "unittest", "discover", "-s", "tests", "-v"],
            output_dir,
        ),
        "local_eval": run_command(
            "local_eval",
            [python, "-X", "utf8", "local_eval.py", "--runs", str(args.runs)],
            output_dir,
        ),
    }
    if not args.skip_benchmark:
        steps["benchmark"] = run_command(
            "benchmark",
            [python, "-X", "utf8", "benchmark.py", "--runs", str(args.runs)],
            output_dir,
        )
    steps["make_submission"] = run_command(
        "make_submission",
        [python, "-X", "utf8", "make_submission.py"],
        output_dir,
    )

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": capture(["git", "rev-parse", "HEAD"]),
            "branch": capture(["git", "branch", "--show-current"]),
            "status_before": git_status_before,
            "status_after": capture(["git", "status", "--porcelain"]) or "",
            "checkout_clean_before": not bool(git_status_before),
        },
        "runtime": {
            "python": sys.version,
            "python_executable": python,
            "platform": platform.platform(),
            "packages": package_versions(),
        },
        "nvidia": {
            "smi": capture(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
        },
        "evaluation_runs": args.runs,
        "steps": steps,
        "submission_sha256": file_sha256(REPO_ROOT / "submission.csv"),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    failed = [name for name, step in steps.items() if step["exit_code"] != 0]
    print(f"Evidence report: {report_path}")
    if report["nvidia"]["smi"]:
        print(f"NVIDIA GPU: {report['nvidia']['smi']}")
    else:
        print("NVIDIA GPU was not detected; run this verifier on a Brev GPU instance.")
    if failed:
        print(f"Failed steps: {', '.join(failed)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
