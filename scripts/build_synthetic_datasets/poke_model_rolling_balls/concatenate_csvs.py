import argparse
import pandas as pd
import os

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Concatenate two CSV files with identical columns.')
    parser.add_argument('--input_path_csv1', required=True, help='Path to the first CSV file')
    parser.add_argument('--input_path_csv2', required=True, help='Path to the second CSV file')
    parser.add_argument('--output_path_csv', required=True, help='Path to save the combined CSV file')
    return parser.parse_args()

def concatenate_csvs(input_path_csv1, input_path_csv2, output_path_csv):
    """Concatenate two CSV files and save the result."""
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the CSV files
    print(f"Reading first CSV from: {input_path_csv1}")
    df1 = pd.read_csv(input_path_csv1)
    print(f"Shape of first CSV: {df1.shape}")
    
    print(f"Reading second CSV from: {input_path_csv2}")
    df2 = pd.read_csv(input_path_csv2)
    print(f"Shape of second CSV: {df2.shape}")
    
    # Check that columns match
    if list(df1.columns) != list(df2.columns):
        print("WARNING: Column names don't match between the two CSV files!")
        print(f"CSV1 columns: {list(df1.columns)}")
        print(f"CSV2 columns: {list(df2.columns)}")
        raise ValueError("CSV files have different columns!")
    
    # Concatenate the dataframes
    combined_df = pd.concat([df1, df2], ignore_index=True)
    print(f"Shape of combined CSV: {combined_df.shape}")
    
    # Save the combined dataframe
    print(f"Saving combined CSV to: {output_path_csv}")
    combined_df.to_csv(output_path_csv, index=False)
    print("Concatenation completed successfully!")

def main():
    args = parse_arguments()
    concatenate_csvs(args.input_path_csv1, args.input_path_csv2, args.output_path_csv)

if __name__ == "__main__":
    main()