import matplotlib.pyplot as plt
import numpy as np
import re
import sys
import os

def parse_results_file(filename):
    """
    Parse the federated learning results file and extract data for plotting.
    
    Args:
        filename (str): Path to the results summary file
        
    Returns:
        tuple: (sigma_values, fedavg, fedavg_jse, scaffold, scaffold_jse, alpha_value)
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Results file '{filename}' not found.")
    
    with open(filename, 'r') as file:
        lines = file.readlines()
    
    # Extract alpha value from the header
    alpha_value = None
    for line in lines:
        if "ALPHA =" in line:
            alpha_match = re.search(r'ALPHA = ([\d.]+)', line)
            if alpha_match:
                alpha_value = alpha_match.group(1)
                break
    
    # Find the data section (skip headers and separators)
    data_lines = []
    start_reading = False
    
    for line in lines:
        line = line.strip()
        # Skip empty lines and separator lines
        if not line or '=' in line or '-' in line or 'sigma' in line.lower():
            continue
        
        # Check if this looks like a data line (starts with a number)
        if re.match(r'^\s*[\d.]+\s+', line):
            data_lines.append(line)
    
    # Parse the data
    sigma_values = []
    fedavg = []
    fedavg_jse = []
    scaffold = []
    scaffold_jse = []
    
    for line in data_lines:
        # Split the line and extract values
        parts = line.split()
        if len(parts) >= 2:  # At least sigma and one algorithm
            try:
                sigma_val = float(parts[0])
                
                # Helper function to parse value or return NaN for 'N/A'
                def parse_value(val_str):
                    val_str = val_str.rstrip('%')
                    if val_str.upper() == 'N/A':
                        return np.nan
                    return float(val_str)
                
                # Parse available columns, fill missing ones with NaN
                fedavg_val = parse_value(parts[1]) if len(parts) > 1 else np.nan
                fedavg_jse_val = parse_value(parts[2]) if len(parts) > 2 else np.nan
                scaffold_val = parse_value(parts[3]) if len(parts) > 3 else np.nan
                scaffold_jse_val = parse_value(parts[4]) if len(parts) > 4 else np.nan
                
                sigma_values.append(sigma_val)
                fedavg.append(fedavg_val)
                fedavg_jse.append(fedavg_jse_val)
                scaffold.append(scaffold_val)
                scaffold_jse.append(scaffold_jse_val)
                
            except ValueError as e:
                print(f"Warning: Skipping line due to parsing error: {line.strip()}")
                continue
    
    return sigma_values, fedavg, fedavg_jse, scaffold, scaffold_jse, alpha_value

def plot_federated_learning_results(filename):
    """
    Create a plot from federated learning results file.
    
    Args:
        filename (str): Path to the results summary file
    """
    try:
        # Parse the data from file
        sigma, fedavg, fedavg_jse, scaffold, scaffold_jse, alpha_value = parse_results_file(filename)
        
        print(f"Successfully loaded data from '{filename}'")
        print(f"Alpha value: {alpha_value}")
        print(f"Number of data points: {len(sigma)}")
        
        # Create the plot
        plt.figure(figsize=(12, 8))
        
        # Convert to numpy arrays for better NaN handling
        sigma = np.array(sigma)
        fedavg = np.array(fedavg)
        fedavg_jse = np.array(fedavg_jse)
        scaffold = np.array(scaffold)
        scaffold_jse = np.array(scaffold_jse)
        
        # Plot lines for each method (only if the algorithm has any non-NaN values)
        if not np.all(np.isnan(fedavg)):
            plt.plot(sigma, fedavg, marker='o', linewidth=2, label='FedAvg', color='blue')
        
        if not np.all(np.isnan(fedavg_jse)):
            plt.plot(sigma, fedavg_jse, marker='s', linewidth=2, label='FedAvg + JSE', color='red')
        
        if not np.all(np.isnan(scaffold)):
            plt.plot(sigma, scaffold, marker='^', linewidth=2, label='SCAFFOLD', color='green')
        
        if not np.all(np.isnan(scaffold_jse)):
            plt.plot(sigma, scaffold_jse, marker='d', linewidth=2, label='SCAFFOLD + JSE', color='orange')
        
        # Customize the plot
        plt.xlabel('Sigma ($\sigma$)', fontsize=14)
        plt.ylabel('Accuracy (%)', fontsize=14)
        
        title = f'Federated Learning Performance vs Noise Level'
        if alpha_value:
            title += f' (Alpha = {alpha_value})'
        plt.title(title, fontsize=16, pad=20)
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Set axis limits for better visualization
        if len(sigma) > 0:
            plt.xlim(0, max(sigma))
        else:
            plt.xlim(0, 1)
        plt.ylim(0, 100)
        
        # Add minor grid
        plt.grid(True, which='minor', alpha=0.2)
        plt.minorticks_on()
        
        # Improve layout
        plt.tight_layout()
        
        # Generate output filename based on input filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        output_filename = f'plot_{base_name}.png'
        
        # Save the plot
        plt.savefig(output_filename, dpi=300, bbox_inches='tight')
        
        # Show the plot
        plt.show()
        
        print(f"Plot saved as '{output_filename}'")
        
    except Exception as e:
        print(f"Error processing file '{filename}': {str(e)}")
        return

# Main execution
if __name__ == "__main__":
    for alpha in [0.01, 0.05, 0.1, 0.3, 0.5]:
        results_filename = f"results_alpha{alpha}_summary.txt"
        plot_federated_learning_results(results_filename) 