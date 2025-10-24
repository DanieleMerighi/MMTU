# MMTU Analysis Scripts

This directory contains analysis scripts for MMTU inference results, optimized for thesis work.

## Main Script: analyze_results.py

**Unified analysis script** that performs complete evaluation and analysis in a single run.

### Usage

```bash
python3 analysis/analyze_results.py <result.jsonl>
```

### Example

```bash
python3 analysis/analyze_results.py inference_results/stratified_dataset.llama-3.2-3b.result.jsonl
```

### What It Does

1. **Re-evaluates** each example using task-specific evaluators to get accurate correctness scores
2. **Computes** precise metrics by table size (small/medium/large)
3. **Analyzes** performance degradation patterns
4. **Identifies** worst/best performing tasks
5. **Generates** threshold recommendations for MCP approach
6. **Produces** ready-to-use CSV files for visualization

### Output

All files are saved to: `results_thesis/<dataset>/<model>/`

#### Generated Files

| File | Description | Use Case |
|------|-------------|----------|
| `accurate_by_size.json` | Precise accuracy by size with statistics | Main metrics for analysis |
| `details_full.csv` | All examples with `correct` column | Per-example analysis, filtering |
| `baseline_analysis.json` | Complete analysis with insights | Comprehensive overview |
| `performance_by_size.csv` | Size breakdown | Quick plotting in Excel/Python |
| `performance_by_task.csv` | Task-level performance | Task comparison |
| `task_size_matrix.csv` | Task × Size matrix | Heatmap visualization |
| `threshold_recommendations.txt` | MCP strategy recommendations | Thesis insights |

### Example Output

```
================================================================================
ANALYSIS SUMMARY
================================================================================

Dataset: stratified_dataset
Model: llama-3.2-3b
Total Examples: 228
Overall Accuracy: 0.1053

--------------------------------------------------------------------------------
PERFORMANCE BY TABLE SIZE
--------------------------------------------------------------------------------

SMALL:
  Count: 119
  Accuracy: 0.1597
  Avg Cells: 70 (range: 3-488)
  Avg Dimensions: 27.9 rows × 4.4 cols

MEDIUM:
  Count: 81
  Accuracy: 0.0617
  Avg Cells: 779 (range: 504-1842)
  Avg Dimensions: 143.9 rows × 20.7 cols

LARGE:
  Count: 28
  Accuracy: 0.0000
  Avg Cells: 3772 (range: 2046-11088)
  Avg Dimensions: 188.6 rows × 53.7 cols

--------------------------------------------------------------------------------
MCP STRATEGY RECOMMENDATIONS
--------------------------------------------------------------------------------
  • Consider MCP approach for medium tables (>488 cells)
  • Consider MCP approach for large tables (>1842 cells)
  • STRONG RECOMMENDATION: Use SQL-based MCP for large tables - performance severely degraded

--------------------------------------------------------------------------------
KEY INSIGHTS FOR THESIS
--------------------------------------------------------------------------------
  • Significant 61.3% performance drop from small to medium tables
  • Significant 100.0% performance drop from medium to large tables
  • Task 'Table-QA' shows severe degradation on large tables - priority for MCP
```

## Features

### 1. Accurate Per-Example Correctness

Unlike `evaluate.py` which may aggregate metrics, this script re-evaluates each example individually:

```python
# Creates effective_correct column that works across all task types:
# - Uses 'correct' (0/1) for most tasks (Table-QA, Data-Imputation, etc.)
# - Uses 'f1' score for Error-Detect, Schema-Matching
# - Computes F1 from precision/recall when needed
```

### 2. Size-Based Analysis

Tables are categorized by total cells (rows × columns):
- **Small:** < 500 cells (actual threshold from data)
- **Medium:** 500-2000 cells
- **Large:** > 2000 cells

### 3. Threshold Detection

Automatically identifies optimal thresholds for MCP strategy switching based on:
- Performance degradation (>15% relative drop)
- Absolute performance on large tables
- Task-specific patterns

### 4. Ready-to-Use Outputs

All CSV files are formatted for immediate use in:
- Python/Pandas analysis
- Excel pivot tables
- Plotting libraries (matplotlib, seaborn)
- Statistical analysis (R, SPSS)

## Analysis Workflow

### Basic Workflow

```bash
# 1. Run inference
python3 inference.py -i datasets/stratified_dataset.jsonl self_deploy --model llama-3.2-3b

# 2. Analyze results (one command!)
python3 analysis/analyze_results.py inference_results/stratified_dataset.llama-3.2-3b.result.jsonl

# 3. View recommendations
cat results_thesis/stratified_dataset/llama-3.2-3b/threshold_recommendations.txt
```

### Multi-Model Comparison

```bash
# Run inference for multiple models
for model in llama-3.2-3b qwen3-8b-awq mistral-7b; do
    python3 inference.py -i datasets/stratified_dataset.jsonl self_deploy --model $model
    python3 analysis/analyze_results.py inference_results/stratified_dataset.$model.result.jsonl
done

# Compare results
python3 -c "
import pandas as pd
import json

models = ['llama-3.2-3b', 'qwen3-8b-awq', 'mistral-7b']
for model in models:
    path = f'results_thesis/stratified_dataset/{model}/accurate_by_size.json'
    with open(path) as f:
        data = json.load(f)
        print(f'{model:20s}: {data[\"overall_accuracy\"]:.2%}')
        for size, metrics in data['overall_by_size'].items():
            print(f'  {size:8s}: {metrics[\"accuracy\"]:.2%}')
"
```

### Custom Analysis

```python
import pandas as pd
import json

# Load detailed results
df = pd.read_csv('results_thesis/stratified_dataset/llama-3.2-3b/details_full.csv')

# Analyze specific task by size
task_df = df[df['task'] == 'Table-QA']
print(task_df.groupby('_size_category')['effective_correct'].agg(['mean', 'count']))

# Find failure patterns
failures = df[df['effective_correct'] == 0]
print(failures['task'].value_counts())

# Cell count vs accuracy
import matplotlib.pyplot as plt
plt.scatter(df['_table_cells'], df['effective_correct'], alpha=0.3)
plt.xlabel('Table Cells')
plt.ylabel('Correctness')
plt.xscale('log')
plt.savefig('accuracy_vs_size.png')
```

## Implementation Details

### Unified Metric System

The script creates an `effective_correct` column that normalizes all metrics:

```python
if 'correct' in df.columns:
    df['effective_correct'] = df['correct']  # Binary 0/1
elif 'f1' in df.columns:
    df['effective_correct'] = df['f1']  # F1 score (0-1)
elif 'precision' and 'recall' in df.columns:
    df['effective_correct'] = 2*p*r/(p+r)  # Compute F1
```

This allows cross-task comparison even when tasks use different metrics.

### Size Metadata Extraction

The script extracts size information from:
1. Top-level columns in result file (preferred)
2. Metadata JSON if not in top-level

This ensures compatibility with different dataset formats.

### Threshold Recommendation Logic

```python
# Flag significant degradation
if (acc_small - acc_medium) / acc_small > 0.15:
    recommend_mcp_for_medium = True

# Strong recommendation for severe degradation
if acc_large < acc_small * 0.5:
    recommend_sql_based_mcp = True
```

## Troubleshooting

### Issue: NL2SQL tasks fail evaluation

**Cause:** SQL evaluation requires database files in `data_sqlite/`

**Solution:** Script automatically skips failed tasks and continues

### Issue: Size categories are empty

**Cause:** Dataset doesn't include `_size_category` metadata

**Solution:** Rebuild dataset or add metadata manually

### Issue: Different metrics for different tasks

**Not an issue!** The `effective_correct` column handles this automatically

## Comparison with Old Workflow

| Feature | Old (evaluate.py) | New (analyze_results.py) |
|---------|------------------|-------------------------|
| Per-example correct | ❌ Aggregated | ✅ Individual scores |
| Unified metrics | ❌ Task-specific | ✅ Normalized to 0-1 |
| Threshold detection | ❌ Manual | ✅ Automatic |
| Size analysis | ⚠️ Approximated | ✅ Precise |
| MCP recommendations | ❌ None | ✅ Generated |
| Ready-to-plot CSV | ❌ Basic | ✅ Comprehensive |
| Single command | ❌ Multiple steps | ✅ One command |

## References

- Main documentation: `WORKFLOW.md` in project root
- Configuration guide: `.claude/CLAUDE.md`
- Evaluator implementations: `evaluators/` directory
