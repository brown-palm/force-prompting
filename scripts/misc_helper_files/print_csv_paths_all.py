import os
import sys
from pathlib import Path

def find_csvs_in_subdirs(base_dir):
    # Convert to Path object for easier manipulation
    base_path = Path(base_dir)
    
    # Check if the provided directory exists
    if not base_path.exists() or not base_path.is_dir():
        print(f"Error: '{base_dir}' is not a valid directory", file=sys.stderr)
        sys.exit(1)
    
    # Get all subdirectories in alphabetical order
    subdirs = sorted([d for d in base_path.iterdir() if d.is_dir()])
    
    # Find all CSV files in each subdirectory
    all_csvs = []
    for subdir in subdirs:
        csvs_in_subdir = sorted([str(f) for f in subdir.glob("**/*.csv")])
        all_csvs.extend(csvs_in_subdir)
    
    # Print in the format needed for bash array
    if all_csvs:
        csv_paths_quoted = [f'"{csv}"' for csv in all_csvs]
        print(f'\n\nIMAGE_CSVS=({" ".join(csv_paths_quoted)})')
        print("\n\nQuantity: ", len(csv_paths_quoted))
    else:
        print('IMAGE_CSVS=()')
        print("# No CSV files found in subdirectories", file=sys.stderr)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <directory_path>", file=sys.stderr)
        sys.exit(1)
    
    find_csvs_in_subdirs(sys.argv[1])