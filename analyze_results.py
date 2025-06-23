#!/usr/bin/env python3


import re
import os
import csv
from pathlib import Path


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
       for line in lines:
           if "==================== RESULTS ====================" in line:
               results_section = True
               continue
          
           if results_section and "accuracy:" in line:
               # Extract accuracy value
               acc_match = re.search(r'accuracy:\s*([0-9.]+)%', line)
               if acc_match:
                   accuracy = float(acc_match.group(1))
                   results[sigma] = accuracy
                   break
  
   return results


def create_results_table():
   """
   Create a comprehensive results table from all experiment files.
   """
  
   # Define the algorithms and their corresponding files
   algorithms = {
       'fedavg': 'fedavg_mnist_experiment_result.txt',
       'fedavg + jse': 'fedavg_mnist_experiment_jse_result.txt',
       'scaffold': 'scaffold_mnist_experiment_result.txt',
       'scaffold + jse': 'scaffold_mnist_experiment_jse_result.txt'
   }
  
   # Expected sigma values
   sigma_values = [0, 0.01, 0.05, 0.1, 0.3, 0.5]
  
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


def print_table(column_names, data):
   """
   Print the results table in a nice format.
   """
   print("\n" + "="*80)
   print("FEDERATED LEARNING EXPERIMENT RESULTS SUMMARY")
   print("="*80)
   print()
  
   # Calculate column widths
   col_widths = []
   for i, col_name in enumerate(column_names):
       max_width = max(len(str(col_name)), max(len(str(row[i])) for row in data))
       col_widths.append(max_width + 2)  # Add padding
  
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
  
   print()


def save_table_to_csv(column_names, data, filename='experiment_results_summary.csv'):
   """
   Save the results table to a CSV file.
   """
   with open(filename, 'w', newline='') as csvfile:
       writer = csv.writer(csvfile)
       writer.writerow(column_names)
       writer.writerows(data)
   print(f"Results saved to: {filename}")


def generate_summary_statistics(all_results):
   """
   Generate some summary statistics from the results.
   """
   print("="*80)
   print("SUMMARY STATISTICS")
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


def main():
   """
   Main function to analyze results and generate table.
   """
   print("Analyzing Federated Learning Experiment Results...")
   print("-" * 50)
  
   # Check if we're in the right directory
   current_files = os.listdir('.')
   expected_files = [
       'fedavg_mnist_experiment_result.txt',
       'fedavg_mnist_experiment_jse_result.txt',
       'scaffold_mnist_experiment_result.txt',
       'scaffold_mnist_experiment_jse_result.txt'
   ]
  
   found_files = [f for f in expected_files if f in current_files]
   missing_files = [f for f in expected_files if f not in current_files]
  
   if missing_files:
       print("Missing result files:")
       for f in missing_files:
           print(f"  - {f}")
       print("\nNote: Table will show 'N/A' for missing experiments.")
       print()
  
   # Create results table
   try:
       column_names, data, all_results = create_results_table()
      
       # Print table
       print_table(column_names, data)
      
       # Save to CSV
       save_table_to_csv(column_names, data)
      
       # Generate summary statistics
       generate_summary_statistics(all_results)
      
   except Exception as e:
       print(f"Error processing results: {str(e)}")
       return 1
  
   return 0


if __name__ == "__main__":
   exit(main())

