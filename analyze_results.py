#!/usr/bin/env python3


import re
import os
import glob
from pathlib import Path
from collections import defaultdict


def extract_results_from_file(filename):
   """
   Extract sigma values and their corresponding accuracy from a result file.
   Returns a dictionary: {sigma: accuracy}
   """
   if not os.path.exists(filename):
       print(f"Warning: File {filename} not found!")
       return {}
  
   results = {}
  
   with open(filename, 'r') as f:
       content = f.read()
  
   # Split content by experiment sections
   experiments = content.split('Running experiment with dp_sigma=')
  
   for exp in experiments[1:]:  # Skip the first empty section
       lines = exp.strip().split('\n')
      
       # Extract sigma value from the first line (which just contains the sigma value)
       if not lines or not lines[0].strip():
           continue
      
       try:
           sigma = float(lines[0].strip())
       except ValueError:
           continue
      
       # Find the RESULTS section for this experiment
       results_section = False
       for i, line in enumerate(lines):
           # Handle multiple formats:
           # 1. "==================== RESULTS ===================="
           # 2. "==================== RESULTS" followed by "===================="
           # 3. "====" followed by "RESULTS" followed by "====" 
           if ("==================== RESULTS" in line or 
               (line.strip() == "RESULTS" or line.strip().startswith("RESULTS "))):
               results_section = True
               continue
          
           # Skip lines that are just "====" (continuation of RESULTS header)
           if results_section and line.strip() and "====" in line and "accuracy:" not in line:
               continue
               
           if results_section and "accuracy:" in line:
               # Extract accuracy value
               acc_match = re.search(r'accuracy:\s*([0-9.]+)%', line)
               if acc_match:
                   accuracy = float(acc_match.group(1))
                   results[sigma] = accuracy
                   break
  
   return results


def discover_alpha_values():
    """
    Discover all alpha values from existing result files.
    Returns a sorted list of alpha values.
    """
    alpha_values = set()
    
    # Look for files matching the pattern: {algorithm}_alpha{alpha}_mnist_experiment{jse_suffix}_result.txt
    pattern = "*_alpha*_mnist_experiment*_result.txt"
    files = glob.glob(pattern)
    
    for filename in files:
        # Extract alpha value from filename
        match = re.search(r'_alpha([0-9.]+)_mnist_experiment', filename)
        if match:
            alpha_values.add(float(match.group(1)))
    
    return sorted(alpha_values)


def get_algorithm_files_for_alpha(alpha):
    """
    Get all algorithm result files for a specific alpha value.
    Returns a dictionary: {algorithm_name: filename}
    """
    algorithms = {}
    
    # Define expected algorithm combinations
    algorithm_patterns = [
        ('fedavg', f'fedavg_alpha{alpha}_mnist_experiment_result.txt'),
        ('fedavg + jse', f'fedavg_alpha{alpha}_mnist_experiment_jse_result.txt'),
        ('scaffold', f'scaffold_alpha{alpha}_mnist_experiment_result.txt'),
        ('scaffold + jse', f'scaffold_alpha{alpha}_mnist_experiment_jse_result.txt')
    ]
    
    for alg_name, filename in algorithm_patterns:
        if os.path.exists(filename):
            algorithms[alg_name] = filename
    
    return algorithms


def create_results_table_for_alpha(alpha):
   """
   Create a results table for a specific alpha value.
   Returns column_names, data, and all_results for that alpha.
   """
   
   # Get algorithm files for this alpha
   algorithms = get_algorithm_files_for_alpha(alpha)
   
   if not algorithms:
       print(f"Warning: No result files found for alpha={alpha}")
       return None, None, None
  
   # Expected sigma values
#    sigma_values = [0, 0.01, 0.03, 0.05, 0.07, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
   sigma_values = [0, 0.01, 0.05, 0.1, 0.3, 0.5, 1]
  
   # Extract results for each algorithm
   all_results = {}
   for alg_name, filename in algorithms.items():
       print(f"Processing {filename}...")
       results = extract_results_from_file(filename)
       all_results[alg_name] = results
       print(f"  Found results for sigmas: {sorted(results.keys())}")
  
   # Create data structure for table
   column_names = ['sigma'] + list(algorithms.keys())
   data = []
  
   for sigma in sigma_values:
       row = [sigma]
       for alg_name in algorithms.keys():
           if sigma in all_results[alg_name]:
               row.append(f"{all_results[alg_name][sigma]:.2f}%")
           else:
               row.append("N/A")
       data.append(row)
  
   return column_names, data, all_results


def print_table_for_alpha(alpha, column_names, data):
   """
   Print the results table for a specific alpha value in a nice format.
   """
   print("\n" + "="*100)
   print(f"FEDERATED LEARNING EXPERIMENT RESULTS - ALPHA = {alpha}")
   print("="*100)
   print()
  
   # Calculate column widths
   col_widths = []
   for i, col_name in enumerate(column_names):
       max_width = max(len(str(col_name)), max(len(str(row[i])) for row in data))
       col_widths.append(max_width + 3)  # Add more padding for better readability
  
   # Print header
   header_line = ""
   for i, col_name in enumerate(column_names):
       header_line += f"{col_name:^{col_widths[i]}}"
   print(header_line)
  
   # Print separator
   separator = ""
   for width in col_widths:
       separator += "-" * width
   print(separator)
  
   # Print data rows
   for row in data:
       row_line = ""
       for i, cell in enumerate(row):
           row_line += f"{str(cell):^{col_widths[i]}}"
       print(row_line)
  
   print("="*100)
   print()


def save_table_to_txt_for_alpha(alpha, column_names, data):
   """
   Save the results table for a specific alpha to a nicely formatted text file.
   """
   filename = f'results_alpha{alpha}_summary.txt'
   
   with open(filename, 'w') as f:
       # Write header
       f.write("="*100 + "\n")
       f.write(f"FEDERATED LEARNING EXPERIMENT RESULTS - ALPHA = {alpha}\n")
       f.write("="*100 + "\n")
       f.write("\n")
       
       # Calculate column widths (same logic as print function)
       col_widths = []
       for i, col_name in enumerate(column_names):
           max_width = max(len(str(col_name)), max(len(str(row[i])) for row in data))
           col_widths.append(max_width + 3)  # Add padding for readability
       
       # Write table header
       header_line = ""
       for i, col_name in enumerate(column_names):
           header_line += f"{col_name:^{col_widths[i]}}"
       f.write(header_line + "\n")
       
       # Write separator
       separator = ""
       for width in col_widths:
           separator += "-" * width
       f.write(separator + "\n")
       
       # Write data rows
       for row in data:
           row_line = ""
           for i, cell in enumerate(row):
               row_line += f"{str(cell):^{col_widths[i]}}"
           f.write(row_line + "\n")
       
       f.write("="*100 + "\n")
   
   print(f"Results table for alpha={alpha} saved to: {filename}")


def generate_summary_statistics_for_alpha(alpha, all_results):
   """
   Generate summary statistics for a specific alpha value.
   """
   print("="*80)
   print(f"SUMMARY STATISTICS FOR ALPHA = {alpha}")
   print("="*80)
  
   for alg_name, results in all_results.items():
       if not results:
           continue
          
       accuracies = list(results.values())
       sigma_vals = list(results.keys())
      
       print(f"\n{alg_name.upper()}:")
       print(f"  Best accuracy: {max(accuracies):.2f}% (σ={min(sigma_vals):.3f})")
       print(f"  Worst accuracy: {min(accuracies):.2f}% (σ={max(sigma_vals):.3f})")
       print(f"  Accuracy drop: {max(accuracies) - min(accuracies):.2f}%")
      
       # Calculate accuracy at specific sigma values
       if 0 in results and 0.1 in results:
           drop_01 = results[0] - results[0.1]
           print(f"  Accuracy drop (σ=0 to σ=0.1): {drop_01:.2f}%")
   
   print("="*80)


def main():
   """
   Main function to analyze results and generate separate tables for each alpha value.
   """
   print("="*100)
   print("FEDERATED LEARNING EXPERIMENT ANALYSIS - ALPHA-SPECIFIC RESULTS")
   print("="*100)
   print("Generating separate tables for each alpha value...")
   print()
  
   # Discover all alpha values
   alpha_values = discover_alpha_values()
   
   if not alpha_values:
       print("ERROR: No result files found with the expected naming pattern!")
       print("Expected pattern: {algorithm}_alpha{alpha}_mnist_experiment{_jse}_result.txt")
       print("Please make sure you have run experiments and have result files in the current directory.")
       return 1
   
   print(f"Found alpha values: {alpha_values}")
   print(f"Will generate {len(alpha_values)} separate tables...")
   print()
   
   # Process each alpha value
   for i, alpha in enumerate(alpha_values, 1):
       print(f"\n{'='*60}")
       print(f"TABLE {i}/{len(alpha_values)}: Processing Alpha = {alpha}")
       print(f"{'='*60}")
       
       # Create results table for this alpha
       column_names, data, all_results = create_results_table_for_alpha(alpha)
       
       if column_names is None:
           print(f"Skipping alpha={alpha} due to missing files.")
           continue
           
       # Print table
       print_table_for_alpha(alpha, column_names, data)
       
       # Save to text
       save_table_to_txt_for_alpha(alpha, column_names, data)
       
       # Generate summary statistics
       generate_summary_statistics_for_alpha(alpha, all_results)
   
   print("\n" + "="*100)
   print("ANALYSIS COMPLETE")
   print("="*100)
   print(f"Generated {len(alpha_values)} separate result tables.")
   print("Text files saved:")
   for alpha in alpha_values:
       print(f"  - results_alpha{alpha}_summary.txt")
   print()
   
   return 0


if __name__ == "__main__":
   exit(main())

