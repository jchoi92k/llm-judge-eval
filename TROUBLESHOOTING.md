# Troubleshooting

## Common Issues

### Directory Structure
Ensure your project structure matches the layout described in the [README.md](README.md#project-structure). Ensure that the following files and folders are in the root directory and not in subdirectories:
- `.env` (containing your OpenAI API key)
- `config_{tool_name}.toml`
- `run_evaluation_{tool_name}.ipynb`
- `data_preprocessing_{tool_name}.ipynb`
- The tool_specific files and utility folders `{tool_name}_files/` and `{tool_name}_utils/`

### Configuration Files
**Example notebooks and config files are placeholders.** Make sure you're using the correct tool-specific versions:
- Use `config_{your_tool_name}.toml`, not the example config
- Use `run_evaluation_{your_tool_name}.ipynb`, not the example notebook

### Session Data Format
If preprocessing fails or evaluations produce unexpected results, verify your `session_data.csv`:
- Must include a `session_id` column (exact name required)
- If using images, the column must be named `image_data_base64`
- Image placeholders in text should follow the format `[Image: 1]`, `[Image: 2]`, etc.
- All columns should be documented in your `session_data_description_{tool_name}.txt` file

### API Key Issues
If you see authentication errors:
```bash
# Verify .env file exists in project root
cat .env  # Linux/Mac
type .env  # Windows

# Should contain:
OPENAI_API_KEY=your_key_here
```

### Kernel/Environment Issues
If the kernel doesn't recognize installed packages:
1. Restart your Jupyter server or refresh the browser
2. Reload VS Code window (Ctrl+Shift+P → "Reload Window")
3. Select the correct kernel from the kernel picker
4. Manually enter the interpreter path if needed (`.venv/Scripts/python.exe` on Windows, `.venv/bin/python` on Mac/Linux)

## Batch Processing

### Check Evaluation Status
```python
evaluator.check_evaluation_status()
```

### Batch Retrieval After Kernel Restart
If your kernel restarts during batch processing, you can resume:
```python
# Find batch_id in logs/{run_id}.log or OpenAI dashboard
evaluator.check_and_retrieve(
    until_complete=True,
    batch_id_override="batch_abc123"
)
```

### Check Batch Status Manually
```python
evaluator.check_batch_status()
```

### Cancel a Batch
```python
evaluator.cancel_batch()
```

## Restarting Evaluations

### Starting Over
To run a completely fresh evaluation:
1. Delete files in `outputs/` directories, or
2. Change output paths in `config_{tool_name}.toml` to use new directories

### Regenerating Guidelines
To regenerate evaluation guidelines without rerunning evaluations:
```python
evaluator.generate_evaluation_guidelines(force_regenerate=True)
```

## Cost and Quota Issues

### Cost Estimates
- Cost estimates are **rough approximations** that only account for text input
- Image inputs are not included in estimates: actual costs may be higher
- Use `auto_approve=False` to review costs before proceeding

### Rate Limits
If you hit rate limits or want faster inference:
- Use `service_tier='default'` for standard pricing with better availability
- Wait and retry—rate limits reset over time

## Output Issues

### Adjudication Not Triggering
Adjudication runs when:
- Any subcriterion has a score gap ≥ 2 between two evaluations, OR
- Evaluations disagree on mathematical relevance

Check `evaluator.check_evaluation_status()` to see if adjudication is needed.

## Model-Specific Issues

### Image Processing Errors
If using images, confirm your model supports vision:
- o3 and o4-mini support images
- Check OpenAI documentation for current model capabilities
- Verify images are properly base64-encoded in `image_data_base64` column

### Model Not Available
If you see model availability errors:
- Verify the `model_name` in your config matches available OpenAI models
- Check your API account has access to the requested model
- Try `service_tier='default'` instead of `'auto'` or `'flex'`; only a number of models allow flex processing.

---

For configuration details, see [CONFIG.md](CONFIG.md).  
For general usage, see [README.md](README.md).