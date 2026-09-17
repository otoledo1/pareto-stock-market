#!/usr/bin/env bash
# Reproduce every table and figure in the paper from the archived panel.
# Step 01 is excluded: it re-pulls from Yahoo Finance and is for independent
# replication only. See README.md.
# Order matters: 13 reads output/tables/exact_conditional_summary.csv from 11.
set -euo pipefail
cd "$(dirname "$0")/src"
for script in 02_build_panel 03_simulate 04_regressions 05_date_clustering \
              06_identity 07_gaussian_benchmark 08_extensions 09_figures \
              10_return_conditioned_full 11_exact_conditional_test \
              12_block_bootstrap 13_correlated_null; do
    echo; echo "=============== ${script} ==============="; echo
    python3 "${script}.py"
done

