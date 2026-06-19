#!/bin/bash

SOURCE_PATH=$1

if [ -z "$SOURCE_PATH" ]; then
    echo "Usage: $0 <source_path>"
    echo "<source_path> is the path to the directory where all results were saved by the benchmark."
    exit 1
fi

CURRENT_PATH=$(pwd)
FOLDERS=("Adult" "ACSIncome" "Compas" "BoD")

for folder in "${FOLDERS[@]}"; do
    SRC="$SOURCE_PATH/$folder"
    if [ -d "$SRC" ]; then
        cp -r "$SRC" "$CURRENT_PATH/"
        echo "Copied: $folder"
    else
        echo "Not found: $SRC"
    fi
done

echo "Done."