# Configuration Guide

Your `config_<tool_name>.toml` file is **pre-configured and ready to use**. This guide explains what each setting does and what you might want to customize.

---

## Quick Customizations

### 1. Sample Size (Most Common Change)

**For testing**, start with a small sample:
```toml
[evaluation_settings]
n_samples = 5  # Test with 5 sessions
```

**For full evaluation**, increase as needed:
```toml
n_samples = 100  # Or your desired sample size
```

### 2. Human Evaluation Examples

Controls how many human evaluation examples are used for: 
- guideline generation
- included as few shot examples in prompts
```toml
n_human_rating_samples = 3
```

If you don't have human evaluations, set to 0 or the pipeline will prompt you.

### 3. Tool Name

Update to match your tutoring tool:
```toml
[tool_settings]
tool_name = "Your Tool Name"  # Change to your tool's name
```

---

## Understanding the Config Sections

### `[evaluation_settings]`

**What it controls:** Sampling and data selection

| Setting | Purpose | Typical Values |
|---------|---------|----------------|
| `n_samples` | How many sessions to evaluate | 5 (testing) - complete set |
| `n_human_rating_samples` | Human examples for guidelines | 3-10 |

**Impact on costs:**
- More `n_samples` = proportionally higher evaluation costs
- More `n_human_rating_samples` = slightly better guidelines (one-time cost)

---

### `[model]`

**What it controls:** Which OpenAI model to use and cost calculations
```toml
model_name = "o3"
price_per_1M_input_tokens = 1.0
price_per_1M_cached_input_tokens = 0.25
price_per_1M_output_tokens = 4.0
expected_output_tokens = 1500
```

**When to change:**
- Switching to a different OpenAI model (update all pricing accordingly)
- OpenAI updates their pricing (check [openai.com/pricing](https://openai.com/pricing))

**Important:** Pricing must match OpenAI's current rates for accurate cost estimates. Flex and batch processing comes with a 50% discount.

---

### `[api_settings]`

**What it controls:** API call behavior and reliability
```toml
max_retries = 3           # Retry failed calls up to 3 times
timeout = 900.0           # Wait up to 15 minutes per call
retry_delay = 2.0         # Wait 2 seconds before retrying
embedding_delay = 0.1     # Delay between embedding calls (rate limiting)
```

**When to change:**
- Experiencing frequent timeouts → increase `timeout`
- Rate limit errors → increase delays
- Want faster failure → decrease `max_retries`

---

### `[tool_settings]`

**What it controls:** Tool-specific configuration
```toml
tool_name = "Your Tool Name"              # Your tutoring tool name
```

**When to change:**
- Different tutoring tool → update `tool_name`
- Different data structure → update `id_column_name`

---

### `[file_paths]`

**What it controls:** Where to find input files

All paths are relative to the project root. **Pre-configured** to match the standard directory structure.
```toml
# Data files
rag_data = "./inputs/RAG/formatted_MRBench.json"
rag_embeddings = "./inputs/RAG/embeddings.pkl"
session_data = "./{team_name}_files/session_data/session_data.csv"
human_evaluation = <not provided>

# Configuration files
evaluation_rubric = "./{team_name}_files/rubrics/{team_name}.json"
session_data_description = "./{team_name}_files/prompt_components/session_data_description_{team_name}.txt"
tool_description = "./{team_name}_files/prompt_components/tool_description_{team_name}.txt"
tool_specific_considerations = "./{team_name}_files/prompt_components/tool_specific_considerations_{team_name}.txt"

# Templates
evaluation_guidelines_template = "./inputs/prompts/evaluation_guideline_generation.j2"
evaluation_template = "./inputs/prompts/evaluation.j2"
evaluation_adjudication_template = "./inputs/prompts/evaluation_adjudication.j2"
evaluation_guidelines_aggregation_template = "./inputs/prompts/evaluation_guidelines_aggregation.j2"
```

**When to change:**
- You've reorganized the directory structure
- Using different input files
- Working with multiple datasets

All files must exist at the specified paths or initialization will fail.

---

### `[dirs]`

```toml
evaluation_guidelines = "./outputs/evaluation_guidelines"
evaluation_results = "./outputs/evaluation_results"
batch_processing = "./outputs/batch_processing"
batch_processing_results = "./outputs/batch_processing_results"
practice_guides = "./inputs/practice_guides"
logs = "./logs"
```

**Auto-created:** These directories are created automatically if they don't exist.

**When to change:**
- Want outputs in a different location
- Sharing outputs across multiple runs

---

## Understanding Run IDs

The pipeline generates a unique `run_id` by hashing:
- All configuration settings
- Session data file **contents**

### What This Means

**Same config + same data = same run_id**
- Can resume interrupted work
- Results append to existing files
- No duplicate evaluations

**Changed config OR data = new run_id**
- Fresh evaluation from scratch
- New output files created
- Old results preserved

**Running new batch with same data and config**
- Delete output files manually

### Example
```
Run 1: n_samples=5, session_data.csv (100 rows)
→ run_id: abc123def456

Run 2: n_samples=10, session_data.csv (100 rows, unchanged)
→ run_id: 789xyz321abc  (different - config changed)

Run 3: n_samples=10, session_data.csv (101 rows, added one)
→ run_id: 456def789ghi  (different - data changed)
```

### Output File Naming

All outputs affixed with run_id with the exception of guidelines:
- `abc123def456_evaluations.pkl`
- `abc123def456_final_scores.json`
- `logs/abc123def456.log`
- `guideline_final.txt`

Guidelines are not tied to specific runs. Delete or force-regenerate when you need new guidelines.

---

## Multiple Configurations

For different evaluation scenarios, create multiple config files:
```
config_{team_name}.toml      # Default/production
config_test.toml             # Small sample for testing
config_research.toml         # Research-specific settings
```

Load specific config:
```python
config = Config.from_toml("config_test.toml")
```

---

## Validation

The pipeline validates your config on load:
- All required fields present
- All file paths exist
- Numeric values in valid ranges
- Directories created automatically

---

## Troubleshooting Config Issues

**"Config file not found"**
→ Ensure `config_{team_name}.toml` is in the same directory (root) as your script/notebook

**"Missing required files"**
→ Check that all paths in `[file_paths]` point to existing files

**"Run ID keeps changing"**
→ Stop editing config or data files mid-run

**"Validation error" with prices**
→ Prices must be positive numbers greater than 0

**Need to reset everything?**
→ Delete files in `outputs/` directories

---

## Getting More Help

- **Usage examples:** See [GUIDE.md](GUIDE.md)
- **Quick start:** See [README.md](README.md)
- **Execution logs:** Check `logs/{run_id}.log`
- **Status check:** Run `print(evaluator)` or `evaluator.check_evaluation_status()`