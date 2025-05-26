import os
import argparse

def find_csv_files(directory_path):
    # List to store the complete paths to all CSV files
    csv_paths = []
    
    # Walk through the directory and find all CSV files
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith('.csv'):
                # Get the complete path to the CSV file
                full_path = os.path.join(root, file)
                csv_paths.append(full_path)
    
    return csv_paths

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Find all CSV files in a directory')
    parser.add_argument('directory', type=str, help='Path to the directory to search for CSV files')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process the directory
    csv_paths = find_csv_files(args.directory)
    
    # Format the output as a bash array
    formatted_paths = '" "'.join(csv_paths)
    
    print(f'\nIMAGE_CSVS=("{formatted_paths}")', "\n")

if __name__ == "__main__":
    main()