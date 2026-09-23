#!/usr/bin/env bash
set -Eeuo pipefail

# Reproducible bootstrap for an NVIDIA Brev VM/Launchable.
# Run from any directory; the script resolves the repository root itself.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${PYTHON_BIN:-python3}"
runs="${BREV_EVAL_RUNS:-10}"

"$python_bin" -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/brev_verify.py --runs "$runs"

echo
echo "Brev verification completed. Evidence: $repo_root/.brev-evidence/report.json"
