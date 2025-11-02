# Evaluation Pipeline

## Overview
An automated evaluation pipeline for assessing LLM-based educational tools, specifically designed for math intervention applications. The pipeline uses OpenAI's API to generate stable, context-aware evaluations with adjudication for disagreements.

**Key Features:**
- Automated evaluation with human-in-the-loop guideline refinement
- Multiple evaluation runs + adjudication system for improved reliability
- Batch/flex processing for reduced cost
- Context augmentation using RAG and practice guides
- Reproducible runs via content-based hashing

## Quick Start
```bash
# Install dependencies on a virtual environment
uv venv .venv
.venv\Scripts\activate.bat # on Windows; or source .venv/bin/activate on Mac
uv pip install -r requirements.txt

# Add API key to .env
echo "OPENAI_API_KEY=your_key_here" > .env
```
Next run the data preprocessing notebook, `data_preprocessing_{tool_name}.ipynb`, followed by the evaluation notebook `notebook evaluation_pipeline_{tool_name}.ipynb`, both found in the tool-specific package.
Refer to the full [Setup section](#setup) or the [Troubleshooting documentation](TROUBLESHOOTING.md) for more troubleshooting or for more information.

## Core Concepts

### Evaluation Flow
1. Load session and human evaluation data
2. Generate evaluation guidelines (3 runs: 2 independent + 1 aggregated)
3. Create context-augmented prompts for all sessions
4. Run evaluations twice per session
5. Identify sessions needing adjudication (score gap ≥2 or mathematical relevance disagreement)
6. Run adjudication on flagged sessions
7. Generate final scores (adjudicated, averaged, or single scores)

### Run Modes
- Full run v. Step-by-step
  - **Full Run**: Single command executes entire pipeline
  - **Step-by-Step**: Manual control through each stage (recommended for first-time users)
- Flex v. Batch
  - **Flex Processing**: 50% cost savings, direct API calls, immediate results
  - **Batch Processing**: 50% cost savings, delayed results (24hr max)

### run_id
Every run generates a unique `run_id` by hashing the config file and session data. All output files are tagged with this ID for reproducibility and version tracking.

## Setup

### Virtual Environment
```bash
uv python install 3.12.10
uv venv .venv
.venv\Scripts\activate.bat # on Windows; or source .venv/bin/activate on Mac
uv pip install -r requirements.txt
uv pip install ipykernel
python -m ipykernel install --user --name=venv --display-name "Evaluator"
```

### API Key
Add your OpenAI API key to `.env`:
```
OPENAI_API_KEY=your_key_here
```

### Data Preparation
1. Use `data_preprocessing_{tool_name}.ipynb` to format your raw data into `session_data.csv`
2. (Optional) Prepare `human_evaluation.csv` for guideline generation and for few-shot examples.
3. Place both files in `inputs/session_data/`

**`session_data.csv` should contain:**
- ID column (column name is required to be `session_id`)
- Interaction data (messages, responses, etc.)
- Any contextual information that helps the LLM understand the session (e.g., student demographics, prior performance, session metadata)
- `image_data_base64` column (if using images)
  - The column name is hard-coded
  - When converting the CSV to a string (with column names as headers and cell contents as values), images will be inserted sequentially into placeholders formatted as `[Image: N]`.
    - Example: if a text cell contains `This is a sample text. The first image: [Image: 1]. The second image: [Image: 2].` and there are two images in the corresponding row in the `image_data_base64` column, they will be inserted in that order.

The schema is flexible—include any number of columns. Document all columns in the `session_data_description` file (specified in your `config_{tool_name}.toml`).

**`human_evaluation.csv` is expected to have:**
- The exact same schema as `session_data.csv`
- Additional columns for each rubric criterion, formatted as `{Domain}_{Subdomain}` (e.g., `Mathematical_Accuracy_Validity`, `Equity_and_Fairness_Cultural_Relevance`)
- Human-assigned scores populating these additional columns

**Note:** Human evaluation data is not provided in this repository or in the team-specific package files. Teams can generate their own human_evaluation.csv through internal human evaluation processes, expert review, or other workflows appropriate to their use case.

### Configuration
Configure all settings in `config_{tool_name}.toml`. Key parameters:
- `n_samples`: Number of sessions to evaluate
- `n_human_rating_samples`: Few-shot examples of human-ratings to include when generating evaluation guidelines as well as in each session evaluation
- `model_name`: Model to use (`o3` or `o4-mini`); confirm that the model can handle image data if your input files contain encoded images.

See [CONFIG.md](CONFIG.md) for full details.

## Usage

### Full Run (Automated)
```python
evaluator = Evaluator(config)
evaluator.run(
    auto_approve=True,
    mode="batch",  # or "flex"
    skip_adjudication=False
)
```

### Step-by-Step Run (Recommended for First Time)
```python
# 1. Generate guidelines
evaluator.generate_evaluation_guidelines()

# 2. Run evaluations (choose flex OR batch)
# Flex:
evaluator.generate_dynamic_prompts()
evaluator.flex_evaluate()
# If the results require adjudication:
evaluator.generate_dynamic_prompts(adjudication=True)
evaluator.flex_evaluate(adjudication=True)

# Batch:
evaluator.generate_dynamic_prompts()
evaluator.prepare_batch_file()
evaluator.upload_batch()
evaluator.check_and_retrieve(until_complete=True)

# 3. Finalize
evaluator.generate_final_scores()
```

## Customization

### Prompts
All prompts use Jinja2 templates in `inputs/prompts/`:
- `evaluation_guideline_generation.j2`
- `evaluation.j2`
- `evaluation_adjudication.j2`
- `evaluation_guidelines_aggregation.j2`

Modify these templates to adjust evaluation behavior. 
For example, if any of the context files have deep nested markdown structures that results in specific, repeated errors where the model confuses injected content, consider adding custom delimiters.
Refer to [this link](https://realpython.com/primer-on-jinja-templating/) for a quick primer on jinja2 templates

### Rubrics
Edit `{tool_name}_files/rubrics/{tool_name}.json` to define your evaluation criteria and scoring system.
The path to the rubric file can also be modified in the config file.

### Tool Descriptions
Customize these plain text files in `{tool_name}_files/prompt_components/`:
- `session_data_description_{tool_name}.txt`: Describe your data format
- `tool_description_{tool_name}.txt`: Overview of the tool being evaluated
- `tool_specific_considerations_{tool_name}.txt`: Special evaluation considerations

### Data Columns
The session data format is flexible. Add or remove columns as needed—just ensure the ID column name matches `config_{tool_name}.toml`.

## Output Files

### Directory Structure
```
outputs/
├── evaluation_guidelines/{run_id}_*.txt
├── evaluation_results/{run_id}_*.json
├── batch_processing/{run_id}_*.jsonl
└── batch_processing_results/{run_id}_*.jsonl
logs/{run_id}.log
```

### Final Scores Format
`outputs/evaluation_results/{run_id}_final_scores.json`:
```json
{
  "session_id_1": {
    "scores": {
      "Mathematical_Accuracy": {
        "Validity": <1-4 or null>,
        "Clarity_and_Labeling": <1-4 or null>,
        "Justification_and_Explanation": <1-4 or null>
      },
      "Pedagogical_Quality": {
        "Problem_Solving_Strategies": <1-4>,
        "Relevance": <1-4>,
        "Scaffolded_Support": <1-4>,
        "Clarity_of_Explanation": <1-4>,
        "Feedback": <1-4>,
        "Motivational_Engagement": <1-4>
      },
      "Equity_and_Fairness": {
        "Language_neutrality": <1-3>,
        "Feedback_tone": <1-3>,
        "Cultural_relevance": <1-3>
      }
    },
    "explanations": {
      "Mathematical_Accuracy": {
        "Validity": "Brief explanation with specific evidence",
        "Clarity_and_Labeling": "Concise justification with examples",
        "Justification_and_Explanation": "Brief reasoning with evidence"
      },
      "Pedagogical_Quality": {
        "Problem_Solving_Strategies": "Brief explanation with evidence",
        "Relevance": "Concise justification",
        "Scaffolded_Support": "Brief reasoning with examples",
        "Clarity_of_Explanation": "Concise explanation",
        "Feedback": "Brief justification",
        "Motivational_Engagement": "Assessment based on student responses when available"
      },
      "Equity_and_Fairness": {
        "Language_neutrality": "Brief explanation",
        "Feedback_tone": "Concise justification",
        "Cultural_relevance": "Brief assessment"
      }
    },
    "mathematical_accuracy_relevance": {
      "applicable": <true/false>,
      "explanation": "Specific analysis of whether AI output contains evaluable mathematical content",
      "extracted_mathematical_content": "If applicable, any mathematical content extracted from the session data by the LLM judges.",
      "catastrophic_errors": "Any significant mathematical errors made by the AI (for example, incorrect calculations such as 2+2=5, or misidentifying a square as a triangle)."
    }
  },
  "session_id_2": {...}
}
```

Adjudications will also include the following field:
```json
  "adjudication_notes": {
    "key_discrepancies_resolved": "Brief summary of main disagreements and how they were resolved",
    "evaluation_preferred": "If one evaluation was generally more accurate, note which (1 or 2) and why"
  }
```

## Other Topics

### RAG Integration
The pipeline uses RAG embeddings (`inputs/RAG/embeddings.pkl`) to retrieve relevant context from a knowledge base (`formatted_MRBench.json`). This augments prompts with domain-specific information. Refer to [Unifying AI Tutor Evaluation: An Evaluation Taxonomy for Pedagogical Ability Assessment of LLM-Powered AI Tutors](https://github.com/kaushal0494/UnifyingAITutorEvaluation) by Kaushal Kumar Maurya, KV Aditya Srivatsa, Kseniia Petukhova, and Ekaterina Kochmar for original data.

### Practice Guides
Math intervention practice guides from Doing What Works (in `inputs/practice_guides/`) are incorporated into evaluation guidelines to ground assessments in evidence-based pedagogy. Refer to this [link](https://ies.ed.gov/ncee/wwc/practiceguides).

### Human Evaluation Format
The `human_evaluation.csv` should provide few-shot examples. Format should match your session data with additional columns for human-assigned scores per rubric criterion.

## Project Structure
```
.
├── run_evaluation_{tool_name}.ipynb         # Main evaluation notebook; provided via download link
├── data_preprocessing_{tool_name}.ipynb     # Data preprocessing notebook - run this first; provided via download link
├── config_{tool_name}.toml                  # Configuration; provided via download link
├── CONFIG.md                                # Config documentation
├── README.md                                # General documentation  
├── inputs/
│   ├── prompts/                             # Jinja2 templates
│   ├── practice_guides/                     # Pedagogy references
│   └── RAG/                                 # Knowledge base + embeddings
├── outputs/                                 # Generated results
├── logs/                                    # Run logs
├── {tool_name}_utils                        # Tool-specific utilities; provided via download link
├── {tool_name}_files                        # Tool-specific files; pprovided via download link
│   ├── rubrics/                
│   ├── session_data/   
│   ├── source_package_data/   
│   └── prompt_components/ 
└── evaluation_pipeline/                     # Main package
```

---

For detailed configuration options, see [CONFIG.md](CONFIG.md).