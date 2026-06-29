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
    echo "  output_suffix: Replaces 'script' in script.out output file"
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
        
        # Set output filename
        if [ -n "$output_suffix" ]; then
            output_file="${output_suffix}.out"
        else
            if [ "${dataset,,}" == "bod" ]; then
                output_file="${script}-combo-${bod_combo}-${number}.out"
            else
                output_file="${script}-${number}.out"
            fi
        fi
        
        echo "Starting: $cmd > $output_file"
        $cmd > "$output_file" 2>&1 &
        wait $!
        echo "Process finished!"
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
            output_file="${output_suffix}.out"
        else
            if [ "${dataset,,}" == "bod" ]; then
                output_file="${script}-combo-${bod_combo}-${number}.out"
            else
                output_file="${script}-${number}.out"
            fi
        fi
        
        echo "Starting: $cmd > $output_file"
        $cmd > "$output_file" 2>&1 &
        wait $! 
        echo "Process finished!"
        ;;
    3)
        # Convert to uppercase
        dataset_upper=$(to_upper "$dataset")

        mkdir -p log
        : > log/$dataset-log-exec-status.log

        # Get systemd scopes whose unit name matches $dataset
        matching_units=$(systemctl --user list-units --type=scope --all --no-legend \
            | awk '{print $1}' | grep -i "$dataset" || true)

        if [[ -z "$matching_units" ]]; then
            echo "No active processes found for ${dataset_upper}" >> log/$dataset-log-exec-status.log
        else
            for unit in $matching_units; do
                echo "*** $unit ***" >> log/$dataset-log-exec-status.log
                systemctl --user show "$unit" \
                    -p ExecMainStartTimestamp -p ActiveState -p SubState \
                    >> log/$dataset-log-exec-status.log

                mem_bytes=$(systemctl --user show "$unit" -p MemoryCurrent --value)
                mem_human=$(numfmt --to=iec --suffix=B "$mem_bytes" 2>/dev/null || echo "N/A")
                echo "Memory=$mem_human" >> log/$dataset-log-exec-status.log

                # PIDs from the cgroup created by scope
                cgroup_path=$(systemctl --user show "$unit" -p ControlGroup --value)
                echo "Control group path: $cgroup_path" >> log/$dataset-log-exec-status.log

                if [[ -n "$cgroup_path" ]]; then
                    pids=$(cat "/sys/fs/cgroup${cgroup_path}/cgroup.procs" 2>/dev/null || true)
                    echo "PIDs in control group: [$pids]" >> log/$dataset-log-exec-status.log

                    if [[ -n "$pids" ]]; then
                        read -r pid cpu mem cmd < <(ps -o pid=,%cpu=,%mem=,cmd= -p $pids)
                        echo "PID: $pid | CPU: ${cpu}% | MEM: ${mem}% | CMD: $cmd" >> log/$dataset-log-exec-status.log 2>/dev/null || true
                    else
                        echo "No active PIDs found in control group for $unit" >> log/$dataset-log-exec-status.log
                    fi
                else
                    echo "No control group path found for $unit " >> log/$dataset-log-exec-status.log
                fi
                echo "" >> log/$dataset-log-exec-status.log
            done

            echo "${dataset_upper} processes under execution" >> log/$dataset-log-exec-status.log
        fi

        cat log/$dataset-log-exec-status.log
        ;;
    *)
        echo "Error: Invalid option. Use 1 or 2"
        exit 1
        ;;
esac
