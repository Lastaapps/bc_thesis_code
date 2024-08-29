#!/bin/bash

echo "Skipping: 
"
NAMES="
test_bench_NAC_colorings_small
test_bench_NAC_colorings_laman_fast
test_bench_NAC_colorings_general_medium
test_bench_NAC_colorings_laman_large_first_n
test_bench_NAC_colorings_general_first_n
"

timestamp=$(date +%s)

OUT_DIR="./benchmarks/results/run_all"
if [ ! -d "$OUT_DIR" ]; then
    mkdir "$OUT_DIR"
fi

for name in $NAMES; do
    pytest benchmarks -vvvsk "$name" |& tee "$OUT_DIR"/"$name"_"$timestamp".log
done
