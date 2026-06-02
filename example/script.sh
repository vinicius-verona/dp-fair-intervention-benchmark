#!/bin/bash

to_lower() {
    echo "$1" | tr '[:upper:]' '[:lower:]'
}

to_upper() {
    echo "$1" | tr '[:lower:]' '[:upper:]'
}

# Default values
option=""
dataset=""
number=""
output_suffix=""
bod_combo=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --option)
            option="$2"
            shift 2
            ;;
        --dataset)
            dataset="$2"
            shift 2
            ;;
        --number)
            number="$2"
            shift 2
            ;;
        --output-suffix)
            output_suffix="$2"
            shift 2
            ;;
        --bod-combo)
            bod_combo="$2"
            shift 2
            ;;
        *)
            # If no flags detected, treat as positional
            if [ -z "$option" ]; then
                option="$1"
            elif [ -z "$dataset" ]; then
                dataset="$1"
            elif [ -z "$number" ]; then
                number="$1"
            elif [ -z "$output_suffix" ]; then
                output_suffix="$1"
            elif [ -z "$bod_combo" ]; then
                bod_combo="$1"
            fi
            shift
            ;;
    esac
done

if [ -z "$option" ] || [ -z "$dataset" ]; then
    echo "Usage: $0 --option <1|2|3> --dataset <dataset> [--number <number>] [--output-suffix <output_suffix>] [--bod-combo <bod_combo>]"
    echo "  1 - Run benchmark script in background"
    echo "  2 - Run data generation script in background"
    echo "  3 - Show processes with dataset name"
    echo ""
    echo "Parameters:"
    echo "  dataset: Adult, ACSI, or Compas"
    echo "  number: Passed to the script being executed -> ex: run-adult.sh 1"
    echo "  output_suffix: Replaces 'script' in nohup-script.out output file"
    echo "  bod_combo: Combo number for BoD dataset (1-6)"
    exit 1
fi

# Validate dataset input in case option chosen is 1 or 2
if [ "$option" -eq 1 ] || [ "$option" -eq 2 ]; then
    case "${dataset,,}" in
        adult|acsi|compas|bod ) ;;
        *)
            echo "Error: Invalid dataset. Please choose Adult, ACSI, Compas, or BoD."
            exit 1
            ;;
    esac

    if [ "${dataset,,}" == "bod" ]; then
        if [ -z "$bod_combo" ]; then
            echo "Error: For BoD dataset, please provide a combo number (1-6)."
            exit 1
        elif ! [[ "$bod_combo" =~ ^[1-6]$ ]]; then
            echo "Error: Invalid combo number for BoD. Please choose a number between 1 and 6."
            exit 1
        fi
    fi
fi

# Execute based on option
case "$option" in
    1)
        # Convert to lowercase
        script=$(to_lower "$dataset")
        
        # Build command with optional number parameter
        if [ -n "$number" ]; then
                if [ "${dataset,,}" == "bod" ]; then
                    cmd="./run-${script}.sh $number $bod_combo"
                else
                    cmd="./run-${script}.sh $number"
                fi
        else
            echo "Error: Invalid option. If executing a script, pass the required parameters (batch | batch + combo)"
            exit 1
        fi
        
        # Determine output filename
        if [ -n "$output_suffix" ]; then
            output_file="nohup-${output_suffix}.out"
        else
            if [ "${dataset,,}" == "bod" ]; then
                output_file="nohup-${script}-combo-${bod_combo}-${number}.out"
            else
                output_file="nohup-${script}-${number}.out"
            fi
        fi
        
        echo "Starting: $cmd > $output_file"
        nohup $cmd > "$output_file" 2>&1 &
        echo "Process started with PID: $!"
        ;;
    2)
        # Convert to lowercase
        script=$(to_lower "$dataset")
        
        # Build command with optional number parameter
        if [ -n "$number" ]; then
                if [ "${dataset,,}" == "bod" ]; then
                    cmd="./run-${script}-generator.sh $number $bod_combo"
                else
                    cmd="./run-${script}-generator.sh $number"
                fi
        else
            echo "Error: Invalid option. If executing a script, pass the required parameters (batch | batch + combo)"
            exit 1
        fi
        
        # Determine output filename
        if [ -n "$output_suffix" ]; then
            output_file="nohup-${output_suffix}.out"
        else
            if [ "${dataset,,}" == "bod" ]; then
                output_file="nohup-${script}-combo-${bod_combo}-${number}.out"
            else
                output_file="nohup-${script}-${number}.out"
            fi
        fi
        
        echo "Starting: $cmd > $output_file"
        nohup $cmd > "$output_file" 2>&1 &
        echo "Process started with PID: $!"
        ;;
    3)
        # Convert to uppercase
        dataset_upper=$(to_upper "$dataset")

        clear && ps -o pid,%cpu,%mem,cmd > log/$dataset-log-exec-status.log && echo "${dataset_upper}-${number}-Benchmark" >> log/$dataset-log-exec-status.log
        ;;
    *)
        echo "Error: Invalid option. Use 1 or 2"
        exit 1
        ;;
esac
