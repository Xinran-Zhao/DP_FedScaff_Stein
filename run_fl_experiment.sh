#!/bin/bash


# Federated Learning MNIST Experiment Script
# This script runs federated learning algorithms with different dp_sigma values and saves all results to a file
# Usage: ./run_fl_experiment.sh <alpha> [algorithm_name] [jse]
# Examples:
#   ./run_fl_experiment.sh 0.05 scaffold
#   ./run_fl_experiment.sh 0.1 scaffold jse
#   ./run_fl_experiment.sh 0.05 fedavg jse


# Function to run data preparation
prepare_data() {
    local alpha=$1
    echo "Preparing data with alpha=$alpha..."
    echo "Command: python ./data/utils/run.py --dataset mnist --root ~/DP_FedScaff_Stein/datasets --alpha $alpha"
    echo "========================================="
    
    python ./data/utils/run.py --dataset mnist --root ~/DP_FedScaff_Stein/datasets --alpha $alpha
    
    if [ $? -eq 0 ]; then
        echo "✓ Data preparation completed successfully"
    else
        echo "✗ Data preparation failed with exit code $?"
        exit 1
    fi
    echo "========================================="
    echo ""
}


# Function to run experiments for a given algorithm and jse setting
run_experiments() {
   local algorithm=$1
   local use_jse=$2
   local jse_param=""
   local jse_suffix=""
  
   if [ "$use_jse" = "true" ]; then
       jse_param="--jse=True"
       jse_suffix="_jse"
   fi
  
   # Validate algorithm file exists
   local algorithm_file="./src/server/${algorithm}.py"
   if [ ! -f "$algorithm_file" ]; then
       echo "Error: Algorithm file ${algorithm_file} not found!"
       return 1
   fi
  
   local output_file="${algorithm}_mnist_experiment${jse_suffix}_result.txt"
   local base_command="python ${algorithm_file} --dataset mnist --clip_bound=100 --global_epochs=100 --local_epochs=20 ${jse_param}"
  
   # Array of dp_sigma values to test
   local dp_sigma_values=(0 0.01 0.05 0.1 0.3 0.5)
  
   # Clear the output file if it exists
   > "$output_file"
  
   local experiment_name="${algorithm^^}"
   if [ "$use_jse" = "true" ]; then
       experiment_name="${experiment_name} + JSE"
   fi
  
   echo "Starting ${experiment_name} MNIST experiments..." | tee -a "$output_file"
   echo "Timestamp: $(date)" | tee -a "$output_file"
   echo "=========================================" | tee -a "$output_file"
  
   # Loop through each dp_sigma value
   for sigma in "${dp_sigma_values[@]}"; do
       echo "" | tee -a "$output_file"
       echo "Running experiment with dp_sigma=$sigma" | tee -a "$output_file"
       echo "Command: $base_command --dp_sigma=$sigma" | tee -a "$output_file"
       echo "----------------------------------------" | tee -a "$output_file"
      
       # Run the command and append output to file
       $base_command --dp_sigma=$sigma 2>&1 | tee -a "$output_file"
      
       # Check if the command was successful
       if [ ${PIPESTATUS[0]} -eq 0 ]; then
           echo "✓ Experiment with dp_sigma=$sigma completed successfully" | tee -a "$output_file"
       else
           echo "✗ Experiment with dp_sigma=$sigma failed with exit code ${PIPESTATUS[0]}" | tee -a "$output_file"
       fi
      
       echo "=========================================" | tee -a "$output_file"
   done
  
   echo "" | tee -a "$output_file"
   echo "All ${experiment_name} experiments completed at $(date)" | tee -a "$output_file"
   echo "Results saved to: $output_file"
   echo ""
}


# Check if alpha parameter is provided (required)
if [ $# -eq 0 ]; then
   echo "Error: Alpha parameter is required!"
   echo "Usage: ./run_fl_experiment.sh <alpha> [algorithm_name] [jse]"
   echo "Examples:"
   echo "  ./run_fl_experiment.sh 0.05 scaffold"
   echo "  ./run_fl_experiment.sh 0.1 scaffold jse"
   echo "  ./run_fl_experiment.sh 0.05 fedavg jse"
   exit 1
fi

# Get alpha parameter (required)
ALPHA=$1

# Prepare data once before running experiments
prepare_data "$ALPHA"

# Check if only alpha is provided - run all combinations
if [ $# -eq 1 ]; then
    echo "Only alpha provided. Running all algorithm combinations..."
    echo "This will run: fedavg, fedavg+jse, scaffold, scaffold+jse"
    echo "============================================================"
    
    run_experiments "fedavg" "false"
    run_experiments "fedavg" "true"
    run_experiments "scaffold" "false"
    run_experiments "scaffold" "true"
    
    echo "All experiment combinations completed!"

# Specific algorithm specified
else
    # Check if algorithm name is provided, default to fedavg
    ALGORITHM=${2:-fedavg}

    # Check if jse parameter is provided
    USE_JSE="false"
    if [ "$3" = "jse" ]; then
        USE_JSE="true"
    fi

    run_experiments "$ALGORITHM" "$USE_JSE"
fi