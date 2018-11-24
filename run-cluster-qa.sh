#!/bin/bash

input=$1  # input folder where JSON dictionaries are
echo "Working with files in folder: $input"

bp_key=$2  # BioPortal API key

timestamp=$(date +%Y%m%d-%H%M%S)

for input_file in $input/*.json; do
    echo
    echo "Processing file: $input_file"

    output_file=$input_file"_qa_$timestamp.csv"
    echo "Outputting results to: $output_file"

    python3 clusterquality.py $input_file $output_file $bp_key
    echo "done"
done