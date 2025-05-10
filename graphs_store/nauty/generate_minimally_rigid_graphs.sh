#!/usr/bin/env bash

# This script generates laman graphs using nauty and nauty laman plugin:
# https://pallini.di.uniroma1.it/
# https://github.com/martinkjlarsson/nauty-laman-plugin

DIR_ALL=minimally_rigid_all
DIR_SOME=minimally_rigid_some
DIR_DEG_3=minimally_rigid_some_degree_3_plus

if [[ -z "$EXECUTABLE" ]]; then
    EXECUTABLE=nauty-laman-plugin/gensparseg
    echo Using generator: ${EXECUTABLE}
fi
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="./graphs_store/nauty"
    echo Using output dir: ${OUTPUT_DIR}
fi

if [[ ! -d "${OUTPUT_DIR}" ]]; then
    mkdir -p "${OUTPUT_DIR}"
fi
for SUBDIR in general_all general degree_3_plus; do
    if [[ ! -d "${OUTPUT_DIR}/${SUBDIR}" ]]; then
        mkdir -p "${OUTPUT_DIR}/${SUBDIR}"
    fi
done

echo Generation all Minimally rigid graphs...
for n in {5..12}
do
    # Minimally rigid graphs (may be spanned by a triangle component)
    "$EXECUTABLE" "$n" -K2 > "${OUTPUT_DIR}/${DIR_ALL}/minimally_rigid_${n}.g6"
done

echo Generation Minimally rigid graphs...
for n in {5..30}
do
    # Minimally rigid graphs (may be spanned by a triangle component)
    "$EXECUTABLE" $n -K2 | head -n 128 > "${OUTPUT_DIR}/${DIR_SOME}/minimally_rigid_${n}.g6"
done

echo Generation Minimally rigid graphs with min degree 3...
for n in {6..17}
do
    # Minimally rigid graphs with min degree 3
    "$EXECUTABLE" $n -K2 -d3 | head -n 128 > "${OUTPUT_DIR}/${DIR_DEG_3}/minimally_rigid_deg_3_plus${n}.g6"
done

