#!/usr/bin/env bash
# Reproduce every table and figure in the paper from the archived panel.
# Step 01 is excluded: it re-pulls from Yahoo Finance and is for independent
# replication only. See README.md.
set -euo pipefail
cd "$(dirname "$0")/src"
for script in 02_build_panel 03_simulate 04_regressions 05_date_clustering \
              06_identity 07_gaussian_benchmark 08_extensions 09_figures; do
    echo; echo "=============== ${script} ==============="; echo
    python3 "${script}.py"
done
