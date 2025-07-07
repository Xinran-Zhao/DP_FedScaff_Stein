#!/bin/bash


# Federated Learning MNIST Experiment Script
# This script runs federated learning algorithms with different dp_sigma values and saves all results to a file
# Usage: ./run_fl_experiment.sh <alpha1,alpha2,alpha3...> [algorithm_name] [jse]
# Examples:
#   ./run_fl_experiment.sh 0.05,0.1,0.3 scaffold
#   ./run_fl_experiment.sh 0.05,0.1 scaffold jse
#   ./run_fl_experiment.sh 0.05,0.1,0.3 fedavg jse
#   ./run_fl_experiment.sh 0.05,0.1,0.3  (runs all combinations for each alpha)


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
   local alpha=$3
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
  
   local output_file="${algorithm}_alpha${alpha}_mnist_experiment${jse_suffix}_result.txt"
   local base_command="PYTHONPATH=. python ${algorithm_file} --dataset mnist --clip_bound=100 --global_epochs=200 --local_epochs=20 --batch_size=64 ${jse_param}"
  
   # Array of dp_sigma values to test
   local dp_sigma_values=(0 0.01 0.05 0.1)
  
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
   echo "Usage: ./run_fl_experiment.sh <alpha1,alpha2,alpha3...> [algorithm_name] [jse]"
   echo "Examples:"
   echo "  ./run_fl_experiment.sh 0.05,0.1,0.3 scaffold"
   echo "  ./run_fl_experiment.sh 0.05,0.1 scaffold jse"
   echo "  ./run_fl_experiment.sh 0.05,0.1,0.3 fedavg jse"
   echo "  ./run_fl_experiment.sh 0.05,0.1,0.3  (runs all combinations for each alpha)"
   exit 1
fi

# Parse alpha values (comma-separated)
ALPHA_LIST=$(echo $1 | tr ',' ' ')

# Parse algorithm and jse parameters
ALGORITHM=${2:-}
USE_JSE="false"
if [ "$3" = "jse" ]; then
    USE_JSE="true"
fi

# Function to run experiments for all alphas
run_experiments_for_alphas() {
    local algorithm=$1
    local use_jse=$2
    
    for alpha in $ALPHA_LIST; do
        echo ""
        echo "========================================="
        echo "Processing alpha=$alpha"
        echo "========================================="
        
        # Prepare data for this alpha
        prepare_data "$alpha"
        
        # Run experiments for this alpha
        run_experiments "$algorithm" "$use_jse" "$alpha"
        
        echo "Completed experiments for alpha=$alpha"
        echo ""
    done
}

# Check if only alphas are provided - run all combinations
if [ $# -eq 1 ]; then
    echo "Only alphas provided. Running all algorithm combinations for each alpha..."
    echo "Alpha values: $ALPHA_LIST"
    echo "This will run: fedavg, fedavg+jse, scaffold, scaffold+jse for each alpha"
    echo "============================================================"
    
    for alpha in $ALPHA_LIST; do
        echo ""
        echo "========================================="
        echo "Processing alpha=$alpha - All Combinations"
        echo "========================================="
        
        # Prepare data for this alpha
        prepare_data "$alpha"
        
        # Run all combinations for this alpha
        run_experiments "fedavg" "false" "$alpha"
        run_experiments "fedavg" "true" "$alpha"
        run_experiments "scaffold" "false" "$alpha"
        run_experiments "scaffold" "true" "$alpha"
        
        echo "Completed all combinations for alpha=$alpha"
        echo ""
    done
    
    echo "All experiment combinations completed for all alphas!"

# Specific algorithm specified
else
    # Default to fedavg if algorithm not specified
    if [ -z "$ALGORITHM" ]; then
        ALGORITHM="fedavg"
    fi
    
    echo "Running $ALGORITHM with JSE=$USE_JSE for alphas: $ALPHA_LIST"
    echo "============================================================"
    
    run_experiments_for_alphas "$ALGORITHM" "$USE_JSE"
    
    echo "All experiments completed for specified algorithm!"
fi