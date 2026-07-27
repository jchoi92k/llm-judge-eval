# evaluation_pipeline/prompts.py

"""
Prompt building and RAG retrieval utilities.
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Any, Union, Optional
from jinja2 import Template
import pandas as pd

from .config import Config
from . import utils
import json
import logging


def build_output_schema_from_rubric(rubric_json_str: str) -> str:
    """
    Generate the JSON output schema example from the rubric JSON.

    Reads the rubric's categories and criteria to produce a schema that matches
    the actual rubric structure, instead of using a hardcoded schema.

    Uses unquoted placeholders (e.g., <1-4>) to signal to the LLM that values
    should be integers, not strings. This matches the original template style.

    Args:
        rubric_json_str: Raw JSON string of the rubric.

    Returns:
        A pseudo-JSON string with unquoted placeholders for score values.
    """
    rubric = json.loads(rubric_json_str)

    # Build scores and explanations sections as raw text lines
    scores_lines = []
    explanations_lines = []

    categories = rubric["rubrics"]
    for cat_idx, category in enumerate(categories):
        cat_name = category["name"]
        cat_key = cat_name.split(":")[0].strip().replace(" ", "_").replace("-", "_")

        # Scores block for this category
        score_entries = []
        expl_entries = []
        for crit_idx, crit in enumerate(category["criteria"]):
            crit_key = crit["criterion"].replace(" ", "_").replace("-", "_")
            rating_keys = sorted(crit["ratings"].keys(), key=int)
            min_score = int(rating_keys[0])
            max_score = int(rating_keys[-1])

            if cat_key == "Mathematical_Accuracy":
                placeholder = f"<{min_score}-{max_score} or null>"
            else:
                placeholder = f"<{min_score}-{max_score}>"

            comma = "," if crit_idx < len(category["criteria"]) - 1 else ""
            score_entries.append(f'      "{crit_key}": {placeholder}{comma}')
            expl_entries.append(f'      "{crit_key}": "Brief explanation with specific evidence"{comma}')

        cat_comma = "," if cat_idx < len(categories) - 1 else ""
        scores_lines.append(f'    "{cat_key}": {{')
        scores_lines.extend(score_entries)
        scores_lines.append(f'    }}{cat_comma}')

        explanations_lines.append(f'    "{cat_key}": {{')
        explanations_lines.extend(expl_entries)
        explanations_lines.append(f'    }}{cat_comma}')

    scores_block = "\n".join(scores_lines)
    explanations_block = "\n".join(explanations_lines)

    return f"""\
{{
  "scores": {{
{scores_block}
  }},
  "explanations": {{
{explanations_block}
  }},
  "mathematical_accuracy_relevance": {{
    "applicable": <true/false>,
    "explanation": "Specific analysis of whether AI output contains evaluable mathematical content",
    "extracted_mathematical_content": "If applicable, extract any mathematical content from the AI's response for evaluation of accuracy by a Math engine.",
    "catastrophic_errors": "If there are any significant mathematical errors made by the AI, list them here with brief explanations. If none, state 'None'."
  }}
}}"""

# ============================================================================
# RAG RETRIEVAL
# ============================================================================

def retrieve_similar(
    query_embedding: List[float], 
    all_embeddings: Dict[str, List[float]], 
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    Retrieve most similar entries using cosine similarity.
    
    Args:
        query_embedding: Query embedding vector
        all_embeddings: Dictionary mapping IDs to embedding vectors
        top_k: Number of top results to return
        
    Returns:
        List of tuples (id, similarity_score) sorted by similarity
    """
    cids = list(all_embeddings.keys())
    embeddings_matrix = np.array(list(all_embeddings.values()))
    query_emb = np.array(query_embedding)
    
    # Compute cosine similarities using numpy
    dot_product = np.dot(embeddings_matrix, query_emb)
    norms = np.linalg.norm(embeddings_matrix, axis=1) * np.linalg.norm(query_emb)
    similarities = dot_product / norms
    
    # Get top-k indices
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # Return results
    results = [(cids[i], similarities[i]) for i in top_indices]
    return results


# ============================================================================
# PROMPT BUILDER CLASS
# ============================================================================

class PromptBuilder:
    """
    Builds prompts for evaluation using Jinja2 templates.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._load_templates()
        self._load_prompt_components()
    
    def _load_templates(self):
        """Load all Jinja2 templates from config paths."""
        self.guideline_template = Template(
            self.config.file_paths.evaluation_guidelines_template.read_text(encoding='utf-8')
        )
        self.evaluation_template = Template(
            self.config.file_paths.evaluation_template.read_text(encoding='utf-8')
        )
        self.adjudication_template = Template(
            self.config.file_paths.evaluation_adjudication_template.read_text(encoding='utf-8')
        )
        self.guidelines_aggregation_template = Template(
            self.config.file_paths.evaluation_guidelines_aggregation_template.read_text(encoding='utf-8')
        )
    
    def _load_prompt_components(self):
        """Load reusable prompt components."""
        with open(self.config.file_paths.session_data_description, 'r', encoding='utf-8') as f:
            self.session_data_description = f.read()
        
        with open(self.config.file_paths.tool_description, 'r', encoding='utf-8') as f:
            self.tool_description = f.read()
        
        with open(self.config.file_paths.tool_specific_considerations, 'r', encoding='utf-8') as f:
            self.tool_specific_considerations = f.read()
        
        with open(self.config.file_paths.evaluation_rubric, 'r', encoding='utf-8') as f:
            self.rubric_json = f.read()

        self.output_schema = build_output_schema_from_rubric(self.rubric_json)
    
    def build_guideline_prompt(
        self, 
        human_evaluation_samples: Optional[pd.DataFrame],
        practice_guide: str
    ) -> str:
        """
        Build prompt for generating evaluation guidelines.
        
        Args:
            human_evaluation_samples: Sample human evaluations for reference (made optional)
            practice_guide: DWW practice guide text retrieved from: https://ies.ed.gov/ncee/wwc/practiceguides
            
        Returns:
            Rendered prompt string
        """

        # Format samples
        combined_samples = ""

        if human_evaluation_samples is None:
            combined_samples = "No human evaluation samples provided.\n\n"
        
        else:
            for i, row in human_evaluation_samples.iterrows():
                tabulated_data = utils.format_any_tabular_data(row, dataset_name=self.config.model.model_name)
                combined_samples += f"### Sample {i+1}\n{tabulated_data}\n\n"
        
        # Render template
        return self.guideline_template.render(
            SOURCE_CONTENT=practice_guide,
            RUBRIC_JSON=self.rubric_json,
            SAMPLE_DATA_CSV=combined_samples,
            COLUMN_EXPL=self.session_data_description,  
            TOOL_OVERVIEW=self.tool_description, 
            TOOL_SPECIFIC_CONSIDERATION=self.tool_specific_considerations,
        )
    
    def build_guidelines_aggregation_prompt(
        self,
        original_prompt: str,
        guideline_outputs: List[str],
    ) -> str:
        """Build prompt for aggregating multiple guideline outputs."""
        return self.guidelines_aggregation_template.render(
            ORIGINAL_PROMPT=original_prompt,
            outputs=guideline_outputs
        )

    def build_evaluation_prompt(
        self,
        row: pd.Series,
        rag_context: str,
        guideline: str,
        human_evaluation_samples: pd.DataFrame = None
    ) -> List[Dict]:
        """
        Build prompt for evaluating a single session.
        
        Args:
            row: Session data row
            rag_context: Retrieved RAG context
            guideline: Evaluation guideline text
            human_evaluation_samples: Human evaluations for reference (optional)

        Returns:
            Multimodal input structure for API
        """
        # Format row data (exclude image_data_base64 from text)
        row_string = utils.format_any_tabular_data(
            row[~row.index.isin(['image_data_base64'])], 
            f"{self.config.model.model_name} Data"
        )

        # Format human evaluation samples if provided
        combined_samples = ""

        if human_evaluation_samples is not None:
            for i, sample_row in human_evaluation_samples.iterrows():
                tabulated_data = utils.format_any_tabular_data(sample_row, dataset_name=self.config.model.model_name)
                combined_samples += f"### Sample {i+1}\n{tabulated_data}\n\n"
            combined_samples = utils.remove_image_markers(combined_samples)

        
        # Render template
        rendered_prompt = self.evaluation_template.render(
            TOOL_OVERVIEW=self.tool_description,
            RUBRICS=self.rubric_json,
            PRACTICE_GUIDE=guideline,
            EXAMPLE_EVALUATIONS=combined_samples,
            EXAMPLE_TUTOR_CONVERSATIONS=rag_context,
            ROW_DATA=row_string,
            COLUMN_EXPL=self.session_data_description,
            SPECIAL_CONSIDERATION=self.tool_specific_considerations,
            OUTPUT_SCHEMA=self.output_schema,
        )
        
        # Get image data
        image_data = row.get('image_data_base64', None)
        
        # Create multimodal input
        return utils.create_input(rendered_prompt, image_data)
    
    def build_adjudication_prompt(
        self,
        row: pd.Series,
        rag_context: str,
        guideline: str,
        evaluation_1: Dict,
        evaluation_2: Dict
    ) -> List[Dict]:
        """
        Build prompt for adjudicating between two evaluations.
        
        Args:
            row: Session data row
            rag_context: Retrieved RAG context
            guideline: Evaluation guideline text
            evaluation_1: First evaluation result
            evaluation_2: Second evaluation result
            
        Returns:
            Multimodal input structure for API
        """
        import json
        
        # Format row data (exclude image_data_base64 from text)
        row_string = utils.format_any_tabular_data(
            row[~row.index.isin(['image_data_base64'])], 
            f"{self.config.model.model_name} Data"
        )
        
        # Render template
        rendered_prompt = self.adjudication_template.render(
            TOOL_OVERVIEW=self.tool_description,
            RUBRICS=self.rubric_json,
            PRACTICE_GUIDE=guideline,
            EXAMPLE_TUTOR_CONVERSATIONS=rag_context,
            ROW_DATA=row_string,
            COLUMN_EXPL=self.session_data_description,
            SPECIAL_CONSIDERATION=self.tool_specific_considerations,
            EVALUATION_1=json.dumps(evaluation_1, indent=2),
            EVALUATION_2=json.dumps(evaluation_2, indent=2),
            OUTPUT_SCHEMA=self.output_schema,
        )
        
        # Get image data
        image_data = row.get('image_data_base64', None)
        
        # Create multimodal input
        return utils.create_input(rendered_prompt, image_data)
    

def debug_file_loading(config: Config):
    """Debug helper to check file loading issues."""
    
    files_to_check = {
        'session_data_description': config.file_paths.session_data_description,
        'tool_description': config.file_paths.tool_description,
        'tool_specific_considerations': config.file_paths.tool_specific_considerations,
        'evaluation_rubric': config.file_paths.evaluation_rubric,
    }
    
    for name, path in files_to_check.items():
        print(f"\n=== {name} ===")
        print(f"Path: {path}")
        print(f"Exists: {path.exists()}")
        print(f"Is file: {path.is_file()}")
        
        if path.exists():
            print(f"Size: {path.stat().st_size} bytes")
            try:
                content = path.read_text(encoding='utf-8')
                print(f"Content length: {len(content)} chars")
                print(f"First 100 chars: {repr(content[:100])}")
            except Exception as e:
                print(f"Error reading: {e}")