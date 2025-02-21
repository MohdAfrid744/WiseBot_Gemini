import pandas as pd

def load_dataset(file_path):
    """Load dataset from a CSV file using pandas."""
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient="records")
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None  # Return None on error
    except pd.errors.EmptyDataError:
        print(f"Error: CSV file is empty: {file_path}")
        return None  # Return None on error
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None  # Return None on error

def load_all_datasets():
    """Load all datasets from CSV files."""
    datasets = {}
    file_mapping = {
        "Bhagavad Gita": "data/bhagavad_gita.csv",
        "Quran": "data/quran.csv",
        "Bible": "data/bible.csv"
    }
    for name, path in file_mapping.items():
        datasets[name] = load_dataset(path)
        if datasets[name] is None:  # Handle file loading errors.
            print(f"Failed to load dataset: {name}")
            return None  # Return None if any dataset fails to load.
    return datasets

