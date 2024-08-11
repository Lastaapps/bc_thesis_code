#!/bin/bash

# This script generates laman graphs using nauty and nauty laman plugin:
# https://pallini.di.uniroma1.it/
# https://github.com/martinkjlarsson/nauty-laman-plugin

if [[ -z "$EXECUTABLE" ]]; then
    EXECUTABLE=../nauty-laman-plugin/gensparseg
    echo Using generator: ${EXECUTABLE}
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="./graphs_store/nauty-laman"
    echo Using output dir: ${OUTPUT_DIR}
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi

echo Generation Laman graphs...
for n in {5..30}
do
    # Laman graphs (may be spanned by a triangle component)
    "$EXECUTABLE" $n -K2 | head -n 128 > "${OUTPUT_DIR}/general/laman_${n}.g6"
done

echo Generation Laman graphs with min degree 3...
for n in {6..17}
do
    # Laman graphs with min degree 3
    "$EXECUTABLE" $n -K2 -d3 | head -n 128 > "${OUTPUT_DIR}/deg_3_plus/laman_deg_3_plus${n}.g6"
done

