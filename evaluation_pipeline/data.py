# evaluation_pipeline/data.py

"""
Data loading and preprocessing utilities.
"""

import pickle
import json
from pathlib import Path
from typing import Tuple
from ast import literal_eval

import pandas as pd

from .config import Config
from . import utils


# ============================================================================
# DATA LOADING
# ============================================================================
def load_session_data(config: Config, data_prep_function) -> pd.DataFrame:
    """
    Load and preprocess session data for evaluation.
    
    Args:
        config: Configuration object
        data_prep_function: Data preparation function to apply
    Returns:
        Preprocessed DataFrame ready for evaluation
    """
    df = pd.read_csv(config.file_paths.session_data)
    if data_prep_function:
        df = data_prep_function(df, config)
    else:
        pass

    # if "image_data_base64" is in columns and if the type of the column is string, apply literal_eval
    if "image_data_base64" in df.columns and df["image_data_base64"].dtype == object:
        df["image_data_base64"] = df["image_data_base64"].apply(
            lambda x: literal_eval(x) if pd.notna(x) else None
        )
    
    # Sample if configured
    if config.evaluation_settings.n_samples:
        df = df.sample(
            n=config.evaluation_settings.n_samples, 
            random_state=42
        ).reset_index(drop=True)
    
    return df

def load_human_evaluation(config: Config) -> pd.DataFrame:
    """
    Load human evaluation data.
    
    Args:
        config: Configuration object
        
    Returns:
        DataFrame with human evaluation data
    """
    human_evaluation = pd.read_csv(config.file_paths.human_evaluation)
    human_evaluation = human_evaluation.sample(n=config.evaluation_settings.n_human_rating_samples, random_state=42).reset_index(drop=True)

    # remove 'image data base64' column if it exists
    if 'image_data_base64' in human_evaluation.columns:
        human_evaluation = human_evaluation.drop(columns=['image_data_base64'])

    return human_evaluation

def load_rag_data(config: Config) -> Tuple[dict, dict]:
    """
    Load RAG dictionary and embeddings.
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (rag_dictionary, rag_embeddings)
    """
    with open(config.file_paths.rag_data, "r") as f:
        rag_dictionary = json.load(f)
    
    with open(config.file_paths.rag_embeddings, 'rb') as f:
        rag_embeddings = pickle.load(f)
    
    return rag_dictionary, rag_embeddings
