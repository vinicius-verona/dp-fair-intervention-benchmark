#!/usr/bin/env bash

# Usage: ./merge_csvs.sh <input_path> <output_file>
# Merges all CSV files found recursively in <input_path> into <output_file>,
# keeping only the header from the first file encountered.

set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "Usage: $0 <input_path> <output_file>"
    exit 1
fi

INPUT_PATH="$1"
OUTPUT_FILE="$2"

if [[ ! -d "$INPUT_PATH" ]]; then
    echo "Error: '$INPUT_PATH' is not a directory or does not exist."
    exit 1
fi

# Collect all CSV files in that directory level only, sorted for deterministic ordering
mapfile -t CSV_FILES < <(find "$INPUT_PATH" -maxdepth 1 -type f -name "*.csv" | sort)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
    echo "No CSV files found in '$INPUT_PATH'."
    exit 1
fi

echo "Found ${#CSV_FILES[@]} CSV file(s):"
for f in "${CSV_FILES[@]}"; do
    echo "  $f"
done

# Write header from first file, then data rows (no header) from all files
FIRST=true
for CSV in "${CSV_FILES[@]}"; do
    if $FIRST; then
        # Take the full first file (header + data)
        cat "$CSV" > "$OUTPUT_FILE"
        FIRST=false
    else
        # Skip the first line (header) and append the rest
        tail -n +2 "$CSV" >> "$OUTPUT_FILE"
    fi
done

TOTAL_LINES=$(wc -l < "$OUTPUT_FILE")
echo ""
echo "Merged into '$OUTPUT_FILE' ($TOTAL_LINES lines total, including header)."