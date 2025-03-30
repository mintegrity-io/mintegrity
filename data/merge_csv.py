import os
import pandas as pd


def merge_csv_files(input_dir='original', output_file='full_dataset.csv'):
    # Ensure the directory exists
    if not os.path.exists(input_dir):
        print(f"Directory '{input_dir}' does not exist")
        return

    # Get all CSV files in the directory
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    if not csv_files:
        print("No CSV files found in the directory")
        return

    # Read and combine all CSV files
    df_list = [pd.read_csv(os.path.join(input_dir, file)) for file in csv_files]
    print(f"Found {len(csv_files)} CSV files to merge")
    combined_df = pd.concat(df_list, ignore_index=True)

    # Save to a new CSV file
    combined_df.to_csv(output_file, index=False)
    print(f"Combined CSV saved as '{output_file}'")


if __name__ == "__main__":
    merge_csv_files()
