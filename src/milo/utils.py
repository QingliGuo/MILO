# milo/utils.py

"""Utility functions for data loading and preparation."""

from pathlib import Path
import pandas as pd
from . import constants

def load_and_prepare_data(file_path: Path) -> pd.DataFrame:
    """Loads and validates the main input data.

    This function reads a CSV or TSV file, automatically detects whether the
    format is 'samples x features' or 'features x samples' by checking for
    the presence of the 83 indel channels, and transforms it into a
    standardized DataFrame.

    Args:
        file_path (Path): The path to the input data file.

    Returns:
        pd.DataFrame: A DataFrame with samples as rows and the 83 indel
                      channels as columns, ready for analysis.

    Raises:
        IOError: If the file cannot be read.
        ValueError: If the file format is invalid or missing required
                    feature channels.
    """
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', index_col=0)
    except Exception as e:
        raise IOError(f"Error reading the input file: {e}")

    required_features_underscore = set(constants.INDEL_CHANNEL_NAMES)
    required_features_colon = {name.replace('_', ':') for name in constants.INDEL_CHANNEL_NAMES}

    if required_features_underscore.issubset(df.columns) or required_features_colon.issubset(df.columns):
        print("Samples x Features format detected.")
        df.rename(columns=lambda c: str(c).replace(':', '_'), inplace=True)
        final_df = df
    elif required_features_underscore.issubset(df.index) or required_features_colon.issubset(df.index):
        print("Features x Samples format detected.")
        final_df = df.T
        final_df.rename(columns=lambda c: str(c).replace(':', '_'), inplace=True)
    else:
        error_message = (
            "Input file format is invalid. All 83 required indel channels were not found.\n\n"
            "Please ensure your file meets one of these criteria:\n"
            "1. The first column contains sample IDs, and the header contains all 83 feature names.\n"
            "2. The header contains sample IDs, and the first column contains all 83 feature names.\n\n"
            "Indel channels names can use ':' or '_' as separators. Extra columns are ignored."
        )
        raise ValueError(error_message)

    final_df = final_df[list(constants.INDEL_CHANNEL_NAMES)]
    final_df.index.name = 'SampleID'
    return final_df


def load_lp_norm_data(file_path: Path) -> pd.DataFrame:
    """Loads, validates, and standardizes coverage and purity data.

    This function reads a CSV or TSV file provided via the --cov_norm
    argument. It searches for 'coverage' and 'purity' columns or rows
    (case-insensitively) and standardizes the format to have samples as
    rows and 'coverage'/'purity' as columns.

    Args:
        file_path (Path): The path to the coverage/purity data file.

    Returns:
        pd.DataFrame: A DataFrame with SampleIDs as the index and 'coverage'
                      and/or 'purity' as columns.

    Raises:
        ValueError: If a 'coverage' column or row cannot be found in the file.
    """
    df = pd.read_csv(file_path, index_col=0)
    
    # Check if 'coverage' is in columns
    for col in df.columns:
        if str(col).lower() == 'coverage':
            rename_map = {col: 'coverage'}
            for p_col in df.columns:
                if str(p_col).lower() == 'purity':
                    rename_map[p_col] = 'purity'
            df.rename(columns=rename_map, inplace=True)
            return df

    # Check if 'coverage' is in the index (transposed format)
    for idx in df.index:
        if str(idx).lower() == 'coverage':
            new_index = {}
            for i in df.index:
                if str(i).lower() in ['coverage', 'purity']:
                    new_index[i] = str(i).lower()
            df.rename(index=new_index, inplace=True)
            df = df.T
            return df

    raise ValueError("The --cov_norm file must contain a 'coverage' column or row.")
