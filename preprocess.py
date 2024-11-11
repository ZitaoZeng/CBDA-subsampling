import pandas as pd
import argparse
import sys

def identify_longitudinal_data_with_delimiters(df, delimiters=[',', '|']):
    """
    Identify columns with longitudinal data based on presence of delimiters within cells.
    Parameters:
    - df: The DataFrame containing the data.
    - delimiters: List of delimiters to check within cell values (e.g., ',' or '|').
    
    Returns:
    - longitudinal_columns: List of column names that represent longitudinal data.
    - non_longitudinal_columns: List of column names that are not longitudinal data.
    """
    def contains_delimiter(cell):
        return any(delim in str(cell) for delim in delimiters)
    
    longitudinal_columns = [
        col for col in df.columns if df[col].apply(contains_delimiter).any()
    ]
    non_longitudinal_columns = list(set(df.columns) - set(longitudinal_columns))
    
    return longitudinal_columns, non_longitudinal_columns

def map_columns_to_feature_names(data, feature_names_path, ordinals_path, case_id, target_id):
    """
    Map columns in `data` to feature names based on ordinals and insert case_id and target_id.
    Parameters:
    - data: The DataFrame whose columns need to be mapped.
    - feature_names_path: Path to the CSV file with feature names.
    - ordinals_path: Path to the CSV file with column ordinals.
    - case_id: Name of the case identifier column.
    - target_id: Name of the target identifier column.
    
    Returns:
    - data: The DataFrame with columns renamed based on feature names, ordinals, case_id, and target_id.
    """
    try:
        # Load feature names and ordinals
        feature_names = pd.read_csv(feature_names_path, header=None)
        ordinals = pd.read_csv(ordinals_path, header=None).squeeze("columns")
        
        # Select feature names based on ordinals for the rest of the columns
        selected_feature_names = feature_names.iloc[ordinals - 1, 0].tolist()  # -1 for zero-based indexing
        
        # Insert case_id and target_id at the beginning of the column list
        final_column_names = [case_id, target_id] + selected_feature_names
        data.columns = final_column_names
        return data
    
    except Exception as e:
        print(f"Error mapping columns to feature names: {e}")
        sys.exit(1)

def identify_longitudinal_data_with_delimiters(df, delimiters=[',', '|']):
    """
    Identify columns with longitudinal data based on presence of delimiters within cells.
    Parameters:
    - df: The DataFrame containing the data.
    - delimiters: List of delimiters to check within cell values (e.g., ',' or '|').
    
    Returns:
    - longitudinal_columns: List of column names that represent longitudinal data.
    - non_longitudinal_columns: List of column names that are not longitudinal data.
    """
    def contains_delimiter(cell):
        return any(delim in str(cell) for delim in delimiters)
    
    longitudinal_columns = [
        col for col in df.columns if df[col].apply(contains_delimiter).any()
    ]
    non_longitudinal_columns = list(set(df.columns) - set(longitudinal_columns))
    
    return longitudinal_columns, non_longitudinal_columns

def process_longitudinal_data(data_path, feature_names_path, ordinals_path, case_id, target_id, delimiters, longitudinal_output, non_longitudinal_output):
    """
    Process the longitudinal data based on provided data, feature names, and ordinals files, and split into longitudinal and non-longitudinal.
    Parameters:
    - data_path: Path to the data file.
    - feature_names_path: Path to the feature names file.
    - ordinals_path: Path to the column ordinals file.
    - case_id: Name of the case identifier column.
    - target_id: Name of the target identifier column.
    - delimiters: List of delimiters to check for in cells.
    - longitudinal_output: Path to save the longitudinal data CSV file.
    - non_longitudinal_output: Path to save the non-longitudinal data CSV file.
    """
    # Load data
    try:
        data = pd.read_csv(data_path, sep='\t', header=None)
    except Exception as e:
        print(f"Error reading data file: {e}")
        sys.exit(1)

    # Map columns to feature names and insert case_id and target_id
    data = map_columns_to_feature_names(data, feature_names_path, ordinals_path, case_id, target_id)

    # Identify longitudinal and non-longitudinal columns
    longitudinal_cols, non_longitudinal_cols = identify_longitudinal_data_with_delimiters(data, delimiters)

    # Split into longitudinal and non-longitudinal DataFrames
    longitudinal_data = data[longitudinal_cols]
    non_longitudinal_data = data[non_longitudinal_cols]

    # Save each DataFrame to its respective CSV file
    longitudinal_data.to_csv(longitudinal_output, index=False)
    non_longitudinal_data.to_csv(non_longitudinal_output, index=False)
    
    print(f"Longitudinal data saved to {longitudinal_output}")
    print(f"Non-longitudinal data saved to {non_longitudinal_output}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Identify and split longitudinal data columns in a dataset.")
    parser.add_argument("data_path", type=str, help="Path to the data file (CSV format)")
    parser.add_argument("feature_names_path", type=str, help="Path to the feature names file (CSV format)")
    parser.add_argument("ordinals_path", type=str, help="Path to the ordinals file that maps columns to features (CSV format)")
    parser.add_argument("case_id", type=str, help="Name of the case identifier column")
    parser.add_argument("target_id", type=str, help="Name of the target identifier column")
    parser.add_argument("--delimiters", type=str, default=",|", help="Delimiters to check for in cells (default: ',|')")
    parser.add_argument("--longitudinal_output", type=str, help="Output CSV file name for longitudinal data")
    parser.add_argument("--non_longitudinal_output", type=str, help="Output CSV file name for non-longitudinal data")
    
    args = parser.parse_args()

    # Run the main processing function with provided arguments
    process_longitudinal_data(
        args.data_path, 
        args.feature_names_path, 
        args.ordinals_path, 
        args.case_id, 
        args.target_id, 
        list(args.delimiters), 
        args.longitudinal_output, 
        args.non_longitudinal_output
    )

if __name__ == "__main__":
    main()
