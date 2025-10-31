# evaluation_pipeline/config.py

from pathlib import Path
from typing import Optional
import tomllib
import hashlib
import json
import logging

from pydantic import BaseModel, Field, field_validator, model_validator


class ModelConfig(BaseModel):
    model_name: str

    # Pricing details; should be greater than 0
    price_per_1M_input_tokens: float = Field(gt=0)
    price_per_1M_cached_input_tokens: float = Field(gt=0)
    price_per_1M_output_tokens: float = Field(gt=0)

    # Expected output tokens for cost estimation; also should be greater than 0
    expected_output_tokens: int = Field(gt=0, default=1500)
    
    @property
    def input_token_price(self) -> float:
        """Price per single input token."""
        return self.price_per_1M_input_tokens / 1_000_000
    
    @property
    def cached_token_price(self) -> float:
        """Price per single cached input token."""
        return self.price_per_1M_cached_input_tokens / 1_000_000
    
    @property
    def output_token_price(self) -> float:
        """Price per single output token."""
        return self.price_per_1M_output_tokens / 1_000_000


class EvaluationSettings(BaseModel):
    """Settings for the evaluation process."""
    n_samples: int = Field(gt=0)
    n_human_rating_samples: int = Field(gt=0)
    
    @field_validator('n_samples')
    @classmethod
    def validate_n_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_samples must be at least 1")
        return v

    @field_validator('n_human_rating_samples')
    @classmethod
    def validate_n_human_rating_samples(cls, v: int) -> int:
        if v < 1:
            raise ValueError("n_human_rating_samples must be at least 1")
        return v

class ToolSettings(BaseModel):
    """Settings specific to the tutoring tool being evaluated."""
    tool_name: str

class FilePaths(BaseModel):
    """All input file paths with validation."""
    # RAG files
    rag_data: Path
    rag_embeddings: Path
    
    # Prompt components
    session_data_description: Path
    tool_description: Path
    tool_specific_considerations: Path
    
    # Rubrics and data
    evaluation_rubric: Path
    session_data: Path
    human_evaluation: Path
    
    # Templates
    evaluation_guidelines_template: Path
    evaluation_template: Path
    evaluation_adjudication_template: Path
    evaluation_guidelines_aggregation_template: Path
    
    @field_validator('*', mode='before')
    @classmethod
    def convert_to_path(cls, v) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v
    
    @model_validator(mode='after')
    def validate_files_exist(self):
        """Validate that all files exist."""
        missing_files = []
        for field_name, file_path in self.model_dump().items():
            # Don't check session_data or human_evaluation existence here
            if field_name in ['session_data', 'human_evaluation']:
                continue
            if not file_path.exists():
                missing_files.append(f"{field_name}: {file_path}")
        
        if missing_files:
            raise ValueError(
                f"Missing required files:\n" + "\n".join(missing_files)
            )
        return self


class Directories(BaseModel):
    """Output directories with auto-creation."""
    evaluation_guidelines: Path
    evaluation_results: Path
    batch_processing: Path
    batch_processing_results: Path
    practice_guides: Path
    logs: Path
    
    @field_validator('*', mode='before')
    @classmethod
    def convert_to_path(cls, v) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v
    
    @model_validator(mode='after')
    def create_directories(self):
        """Create all directories if they don't exist."""
        for dir_path in self.model_dump().values():
            dir_path.mkdir(parents=True, exist_ok=True)
        return self

class APISettings(BaseModel):
    """Settings for OpenAI API calls."""
    max_retries: int = Field(default=3, ge=1, le=10)
    timeout: float = Field(default=900.0, gt=0)
    retry_delay: float = Field(default=2.0, ge=0)
    embedding_delay: float = Field(default=1.0, ge=0)
    
    @field_validator('max_retries')
    @classmethod
    def validate_retries(cls, v: int) -> int:
        if v < 1:
            raise ValueError("max_retries must be at least 1")
        return v

class Config(BaseModel):
    """Main configuration class for the evaluation pipeline."""
    evaluation_settings: EvaluationSettings
    model: ModelConfig
    tool_settings: ToolSettings
    api_settings: APISettings 
    file_paths: FilePaths
    dirs: Directories
    
    @classmethod
    def from_toml(cls, toml_path: str | Path) -> "Config":
        """Load configuration from a TOML file."""
        toml_path = Path(toml_path)
        if not toml_path.exists():
            raise FileNotFoundError(f"Config file not found: {toml_path}")
        
        with open(toml_path, "rb") as f:
            config_dict = tomllib.load(f)
        
        return cls(**config_dict)

    def validate_session_data(self) -> None:
        """
        Validate session_data file for main notebook usage.
        Checks that required columns exist and data is properly formatted.
        """
        import pandas as pd
        import ast
        
        if not self.file_paths.session_data.exists():
            raise ValueError(
                f"session_data file not found: {self.file_paths.session_data}"
            )
        
        df = pd.read_csv(self.file_paths.session_data)
        
        # Check for required session_id column
        if 'session_id' not in df.columns:
            raise ValueError(
                "session_data must contain 'session_id' column. "
                f"Found columns: {', '.join(df.columns)}"
            )
        
        # Check for duplicate session_ids
        if df['session_id'].duplicated().any():
            duplicates = df[df['session_id'].duplicated(keep=False)]['session_id'].tolist()
            raise ValueError(
                f"Duplicate session_ids found: {duplicates[:5]}{'...' if len(duplicates) > 5 else ''}"
            )
        
        # Check for null or empty session_ids
        if df['session_id'].isnull().any() or (df['session_id'] == '').any():
            raise ValueError("session_id column contains null or empty values")
        
        # Validate image_data_base64 if it exists
        if 'image_data_base64' in df.columns:
            non_null_values = df['image_data_base64'].dropna()
            sample_size = min(10, max(1, len(non_null_values) // 10))
            sample_items = non_null_values.sample(n=sample_size, random_state=42)

            # try applying ast.literal_eval to all sample items
            try:
                parsed_items = sample_items.apply(ast.literal_eval)
            except (ValueError, SyntaxError) as e:
                raise ValueError(
                    f"image_data_base64 column must contain a Python list of strings. Each string should be either a base64-encoded image or a URL. The value must be parseable by ast.literal_eval()"
                    f"\nExample: \"['data:image/png;base64,iVBORw0KG...', 'https://example.com/image.jpg']\"\n"
                    f"Error: {e}"
                )

            # Validate all parsed items
            for idx, val in parsed_items.items():
                if not isinstance(val, list):
                    raise ValueError(
                        f"image_data_base64 column at row {idx} must contain a list. Found: {type(val).__name__}"
                    )
                for item in val:
                    if item is not None and not isinstance(item, str):
                        raise ValueError(
                            f"image_data_base64 column at row {idx} must contain a list of strings or None. Found: {type(item).__name__}"
                        )
                    
                    # check if string starts with 'data:image/' or contains http or https; don't raise an error but log a warning
                    # skip rest of checks if one offender is found
                    if isinstance(item, str):
                        if not (item.startswith('data:image/') or item.startswith('http://') or item.startswith('https://')):
                            logging.warning(
                                f"image_data_base64 column at row {idx} contains an item that does not start with 'data:image/' or 'http(s)://'. Item: {item[:30]}..."
                                f"\nThis may lead to errors during evaluation if the model expects image data or URLs."
                            )

    def to_dict(self) -> dict:
        """Export configuration as a dictionary with string paths."""
        return {
            "evaluation_settings": self.evaluation_settings.model_dump(),
            "model": self.model.model_dump(),
            "tool_settings": self.tool_settings.model_dump(),
            "file_paths": {k: str(v) for k, v in self.file_paths.model_dump().items()},
            "dirs": {k: str(v) for k, v in self.dirs.model_dump().items()},
        }
    
    @property
    def run_id(self) -> str:
        """
        Generate a unique run ID based on configuration and session data contents.
        
        If config settings OR session data file contents change, run_id changes.
        All files generated will use this run_id as an affix.
        """
        import hashlib
        
        config_dict = self.to_dict()
        
        file_hash = hashlib.md5()
        with open(self.file_paths.session_data, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                file_hash.update(chunk)
        
        hash_input = {
            "config": config_dict,
            "session_data_hash": file_hash.hexdigest()
        }
        
        hash_str = json.dumps(hash_input, sort_keys=True)
        return hashlib.md5(hash_str.encode()).hexdigest()[:12]