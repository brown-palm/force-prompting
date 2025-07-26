import os
import csv
import glob

def process_csvs_in_directories(directories):
    # force_values = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    force_values = [0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    
    for directory in directories:
        # Find all CSV files in the directory
        csv_files = glob.glob(os.path.join(directory, "*.csv"))
        
        for csv_file in csv_files:

            new_filename = csv_file # overwrite the original file
            
            try:
                with open(csv_file, 'r', newline='') as infile:
                    reader = csv.reader(infile)
                    
                    # Read the first row (header)
                    try:
                        header = next(reader)
                    except StopIteration:
                        print(f"Warning: {csv_file} appears to be empty. Skipping.")
                        continue
                    
                    # Find the index of the "force" column
                    try:
                        force_index1 = header.index("force")
                        force_index2 = header.index("force_obj2")
                    except ValueError:
                        print(f"Warning: No 'force' column found in {csv_file}. Skipping.")
                        continue
                    
                    # Read the first data row
                    try:
                        first_row = next(reader)
                    except StopIteration:
                        print(f"Warning: {csv_file} has a header but no data rows. Skipping.")
                        continue
                    
                    # Create the output file with modified rows
                    with open(new_filename, 'w', newline='') as outfile:
                        writer = csv.writer(outfile)
                        
                        # Write the header
                        writer.writerow(header)
                        
                        # Write the original first row
                        writer.writerow(first_row)
                        
                        # Write 7 duplicated rows with different force values
                        for force_value in force_values:
                            modified_row = first_row.copy()
                            modified_row[force_index1] = force_value
                            modified_row[force_index2] = force_value
                            writer.writerow(modified_row)
                
                print(f"Successfully processed {csv_file} -> {new_filename}")
                
            except Exception as e:
                print(f"Error processing {csv_file}: {str(e)}")

if __name__ == "__main__":
    # Hardcoded list of directories to process
    
    directories = [
        "datasets/point-force/test/benchmark_multipoke_mass_understanding_quant"
    ]

    process_csvs_in_directories(directories)