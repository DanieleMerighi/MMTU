#!/usr/bin/env python3
"""
Unified MMTU Results Analysis Script

This script performs complete analysis of MMTU inference results:
1. Re-evaluates each example to get accurate correctness scores
2. Computes precise by-size metrics (small/medium/large)
3. Analyzes performance degradation and failure patterns
4. Generates threshold recommendations for MCP approach
5. Produces ready-to-use CSV files and visualizations

Usage:
    python3 analysis/analyze_results.py <result.jsonl>

Example:
    python3 analysis/analyze_results.py inference_results/stratified_dataset.llama-3.2-3b.result.jsonl

Output:
    All files saved to: results_thesis/<dataset>/<model>/
    - accurate_by_size.json           # Precise accuracy by size
    - details_full.csv                # Full details with correct column
    - baseline_analysis.json          # Complete analysis with insights
    - performance_by_size.csv         # Ready for plotting
    - performance_by_task.csv         # Task breakdown
    - task_size_matrix.csv            # Task × Size matrix
    - threshold_recommendations.txt   # MCP strategy recommendations
"""

import json
import os
import sys
from pathlib import Path
from collections import defaultdict
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Set MMTU_HOME if not already set
if 'MMTU_HOME' not in os.environ:
    # Auto-detect MMTU_HOME as the directory containing this script's parent
    script_dir = os.path.dirname(os.path.abspath(__file__))
    mmtu_home = os.path.dirname(script_dir)
    os.environ['MMTU_HOME'] = mmtu_home
    print(f"ℹ️  Auto-detected MMTU_HOME: {mmtu_home}")

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all evaluators
from evaluators.nl2sql import NSEvaluator
from evaluators.tableqa_evaluator import TQAEvaluator
from evaluators.tfv_evaluator import TFVEvaluator
from evaluators.ed_evaluator import EDEvaluator
from evaluators.data_transform_pbe_python_evaluator import DTPBEPythonEvaluator
from evaluators.em_evaluator import EMEvaluator
from evaluators.tniah_evaluator import TNIAHEvaluator
from evaluators.tablelocate_evaluator import TableLocateEvaluator
from evaluators.sm_evaluator import SMEvaluator
from evaluators.data_transform_reshape_evaluator import DataTransformReshapeEvaluator
from evaluators.data_imputation_evaluator import DataImputationEvaluator
from evaluators.list_to_table_evaluator import ListToTableEvaluator
from evaluators.formula_context_evaluator import FormulaPredictContextEvaluator
from evaluators.transform_by_output_target_schema_evaluator import TransformByTargetSchemaEvaluator
from evaluators.transform_by_input_output_evaluator import TransformByInputOutputEvaluator
from evaluators.semantic_transform_evaluator import SemanticTransformEvaluator
from evaluators.semantic_join_evaluator import SemanticJoinEvaluator
from evaluators.header_value_match_evaluator import HeaderValueMatchEvaluator
from evaluators.ar_evaluator import AREvaluator
from evaluators.fd_evaluator import FDEvaluator
from evaluators.sr_evaluator import SREvaluator
from evaluators.cta_evaluator import CTAEvaluator
from evaluators.cea_evaluator import CEAEvaluator
from evaluators.cpa_evaluator import CPAEvaluator
from evaluators.ejd_evaluator import EquiJoinDetectEvaluator

# Evaluator registry
EVALUATOR_DICT = {
    "NL2SQL": NSEvaluator(),
    "Table-QA": TQAEvaluator(),
    "Table-Fact-Verification": TFVEvaluator(),
    "Error-Detect": EDEvaluator(),
    "Data-transform-pbe": DTPBEPythonEvaluator(),
    "Entity-Matching": EMEvaluator(),
    "Table-needle-in-a-haystack": TNIAHEvaluator(),
    "Table-Locate-by-Row-Col": TableLocateEvaluator(),
    "Schema-Matching": SMEvaluator(),
    "Data-transform-reshape": DataTransformReshapeEvaluator(),
    "Data-Imputation": DataImputationEvaluator(),
    "List-to-table": ListToTableEvaluator(),
    "Formula-prediction-context": FormulaPredictContextEvaluator(),
    "Transform-by-output-target-schema": TransformByTargetSchemaEvaluator(),
    "Transform-by-input-output-table": TransformByInputOutputEvaluator(),
    "semantic-transform": SemanticTransformEvaluator(),
    "semantic-join": SemanticJoinEvaluator(),
    "header-value-matching": HeaderValueMatchEvaluator(),
    "Arithmetic-Relationship": AREvaluator(),
    "Functional-Dependency": FDEvaluator(),
    "String-Relationship": SREvaluator(),
    "Cell-entity-annotation": CEAEvaluator(),
    "Column-type-annotation": CTAEvaluator(),
    "Columns-property-anotation": CPAEvaluator(),
    "equi-join-detect": EquiJoinDetectEvaluator(),
}


def extract_dataset_and_model(result_file):
    """Extract dataset and model name from result file path."""
    basename = os.path.basename(result_file)

    if basename.endswith('.result.jsonl'):
        basename = basename[:-len('.result.jsonl')]

    parts = basename.split('.', 1)

    if len(parts) == 2:
        dataset, model = parts
        return dataset, model
    elif len(parts) == 1:
        return parts[0], 'unknown_model'

    return 'unknown_dataset', 'unknown_model'


def compute_effective_correct(df):
    """Compute unified 'effective_correct' score from various metrics."""
    if 'correct' in df.columns:
        df['effective_correct'] = df['correct']
    elif 'f1' in df.columns:
        df['effective_correct'] = df['f1']
    elif 'precision' in df.columns and 'recall' in df.columns:
        df['effective_correct'] = 2 * df['precision'] * df['recall'] / (df['precision'] + df['recall'])
        df['effective_correct'] = df['effective_correct'].fillna(0)
    elif 'acc' in df.columns:
        df['effective_correct'] = df['acc']
    else:
        df['effective_correct'] = 0

    df['effective_correct'] = pd.to_numeric(df['effective_correct'], errors='coerce').fillna(0)
    return df


def reevaluate_results(result_file):
    """
    Re-evaluate inference results to get accurate per-example correctness.
    Returns DataFrame with all results and size metadata.
    """
    print("=" * 80)
    print("STEP 1: RE-EVALUATING INFERENCE RESULTS")
    print("=" * 80)

    print(f"\nLoading: {result_file}")
    df = pd.read_json(result_file, lines=True)
    print(f"Loaded {len(df)} examples")

    if len(df) == 0:
        print("ERROR: Empty result file")
        return None

    # Ensure task column exists
    if "task" not in df.columns:
        if "metadata" in df.columns:
            print("Extracting task from metadata...")
            df["task"] = df["metadata"].apply(
                lambda x: json.loads(x)["task"] if isinstance(x, str) else x["task"]
            )
        else:
            print("ERROR: No task column and no metadata column")
            return None

    all_results = []

    print("\nProcessing by task...")
    print("-" * 80)

    for task, task_df in df.groupby("task"):
        print(f"\n[{task}] Processing {len(task_df)} examples...")

        if task not in EVALUATOR_DICT:
            print(f"  ⚠ No evaluator found for task: {task}")
            continue

        evaluator = EVALUATOR_DICT[task]

        try:
            task_df_reset = task_df.reset_index(drop=True)
            result_df = evaluator.parse_raw_result(task_df_reset, n_jobs=1)

            # Add size metadata from original task_df
            size_cols = ['_size_category', '_table_rows', '_table_cols', '_table_cells']

            for col in size_cols:
                if col in task_df_reset.columns:
                    result_df[col] = task_df_reset[col].values
                else:
                    def extract_from_metadata(metadata_str):
                        try:
                            metadata = json.loads(metadata_str) if isinstance(metadata_str, str) else metadata_str
                            return metadata.get(col)
                        except:
                            return None

                    result_df[col] = task_df_reset['metadata'].apply(extract_from_metadata).values

            print(f"  ✓ Evaluated {len(result_df)} examples")
            all_results.append(result_df)

        except Exception as e:
            print(f"  ⚠ Error evaluating task {task}: {str(e)[:100]}")
            continue

    if not all_results:
        print("\nERROR: No results to aggregate")
        return None

    all_df = pd.concat(all_results, axis=0, ignore_index=True)
    print(f"\n✓ Total evaluated: {len(all_df)} examples")

    # Compute effective_correct for all rows
    all_df = compute_effective_correct(all_df)

    return all_df


def compute_size_metrics(df, size_col='_size_category', correct_col='effective_correct'):
    """Compute accuracy metrics by size category."""
    metrics = {}

    if size_col not in df.columns:
        return metrics

    if correct_col not in df.columns:
        df = compute_effective_correct(df)

    if not pd.api.types.is_numeric_dtype(df[correct_col]):
        return metrics

    for size_cat in ['small', 'medium', 'large']:
        size_df = df[df[size_col] == size_cat]
        if len(size_df) > 0:
            metrics[size_cat] = {
                'count': int(len(size_df)),
                'total_score': float(size_df[correct_col].sum()),
                'avg_score': float(size_df[correct_col].mean()),
                'accuracy': float(size_df[correct_col].mean()),  # Alias for compatibility
            }

            # Add dimension statistics
            if '_table_cells' in size_df.columns:
                metrics[size_cat]['avg_cells'] = float(size_df['_table_cells'].mean())
                metrics[size_cat]['min_cells'] = int(size_df['_table_cells'].min())
                metrics[size_cat]['max_cells'] = int(size_df['_table_cells'].max())

            if '_table_rows' in size_df.columns:
                metrics[size_cat]['avg_rows'] = float(size_df['_table_rows'].mean())
                metrics[size_cat]['min_rows'] = int(size_df['_table_rows'].min())
                metrics[size_cat]['max_rows'] = int(size_df['_table_rows'].max())

            if '_table_cols' in size_df.columns:
                metrics[size_cat]['avg_cols'] = float(size_df['_table_cols'].mean())
                metrics[size_cat]['min_cols'] = int(size_df['_table_cols'].min())
                metrics[size_cat]['max_cols'] = int(size_df['_table_cols'].max())

    return metrics


def analyze_by_task_and_size(df):
    """Analyze performance matrix: task × size."""
    if df is None or df.empty:
        return {}

    if '_size_category' not in df.columns or 'task' not in df.columns:
        return {}

    task_size_analysis = {}

    for task in df['task'].unique():
        task_df = df[df['task'] == task]
        task_size_analysis[task] = compute_size_metrics(task_df)

    return task_size_analysis


def identify_failure_patterns(df):
    """Identify worst performing tasks and sizes."""
    if df is None or df.empty:
        return {}

    failures = {
        'worst_performing_tasks': [],
        'worst_performing_sizes': [],
        'best_performing_tasks': [],
    }

    # Tasks by accuracy
    if 'task' in df.columns and 'effective_correct' in df.columns:
        task_accs = df.groupby('task')['effective_correct'].mean().sort_values()

        failures['worst_performing_tasks'] = [
            {'task': task, 'accuracy': float(acc)}
            for task, acc in task_accs.head(5).items()
        ]

        failures['best_performing_tasks'] = [
            {'task': task, 'accuracy': float(acc)}
            for task, acc in task_accs.tail(5).items()
        ]

    # Sizes by accuracy
    if '_size_category' in df.columns and 'effective_correct' in df.columns:
        size_accs = df.groupby('_size_category')['effective_correct'].mean()

        failures['worst_performing_sizes'] = [
            {'size': size, 'accuracy': float(acc)}
            for size, acc in size_accs.sort_values().items()
        ]

    return failures


def compute_threshold_recommendations(size_analysis, task_size_analysis):
    """Generate threshold recommendations for MCP strategy switching."""
    recommendations = {
        'cell_count_thresholds': {},
        'strategy_recommendations': [],
        'key_insights': [],
    }

    if not size_analysis:
        return recommendations

    # Extract accuracies by size
    sizes = ['small', 'medium', 'large']
    size_data = []

    for size in sizes:
        if size in size_analysis:
            acc = size_analysis[size].get('accuracy', 0)
            max_cells = size_analysis[size].get('max_cells')
            size_data.append((size, acc, max_cells))

    # Analyze performance degradation
    if len(size_data) >= 2:
        for i in range(len(size_data) - 1):
            size1, acc1, max1 = size_data[i]
            size2, acc2, max2 = size_data[i + 1]

            if acc1 > 0:  # Avoid division by zero
                degradation = (acc1 - acc2) / acc1

                if degradation > 0.15:  # >15% relative drop
                    recommendations['key_insights'].append(
                        f"Significant {degradation:.1%} performance drop from {size1} to {size2} tables"
                    )
                    recommendations['strategy_recommendations'].append(
                        f"Consider MCP approach for {size2} tables (>{max1} cells)" if max1 else
                        f"Consider MCP approach for {size2} tables"
                    )

    # Set threshold values
    if 'small' in size_analysis and 'max_cells' in size_analysis['small']:
        recommendations['cell_count_thresholds']['small_to_medium'] = size_analysis['small']['max_cells']

    if 'medium' in size_analysis and 'max_cells' in size_analysis['medium']:
        recommendations['cell_count_thresholds']['medium_to_large'] = size_analysis['medium']['max_cells']

    # Overall strategy recommendation
    if size_data:
        small_acc = size_data[0][1] if len(size_data) > 0 else 0
        large_acc = size_data[-1][1] if len(size_data) > 0 else 0

        if large_acc < small_acc * 0.5:  # >50% degradation
            recommendations['strategy_recommendations'].append(
                "STRONG RECOMMENDATION: Use SQL-based MCP for large tables - performance severely degraded"
            )
        elif large_acc < small_acc * 0.8:  # >20% degradation
            recommendations['strategy_recommendations'].append(
                "RECOMMENDATION: Consider hybrid MCP approach for medium/large tables"
            )
        else:
            recommendations['strategy_recommendations'].append(
                "Direct inclusion viable - performance remains stable across sizes"
            )

    # Task-specific recommendations
    if task_size_analysis:
        for task, task_sizes in task_size_analysis.items():
            if 'small' in task_sizes and 'large' in task_sizes:
                small_acc = task_sizes['small'].get('accuracy', 0)
                large_acc = task_sizes['large'].get('accuracy', 0)

                if small_acc > 0 and large_acc < small_acc * 0.5:
                    recommendations['key_insights'].append(
                        f"Task '{task}' shows severe degradation on large tables - priority for MCP"
                    )

    return recommendations


def save_all_results(df, size_analysis, task_size_analysis, failure_patterns,
                     threshold_recs, dataset_name, model_name, result_file):
    """Save all analysis results to output directory."""
    output_dir = Path("results_thesis") / dataset_name / model_name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # 1. Accurate by size JSON
    accurate_by_size = {
        'timestamp': datetime.now().isoformat(),
        'source_file': result_file,
        'dataset': dataset_name,
        'model': model_name,
        'total_examples': len(df),
        'overall_accuracy': float(df['effective_correct'].mean()),
        'overall_by_size': size_analysis,
        'by_task_and_size': task_size_analysis,
    }

    with open(output_dir / 'accurate_by_size.json', 'w') as f:
        json.dump(accurate_by_size, f, indent=2)
    print(f"✓ Saved: {output_dir / 'accurate_by_size.json'}")

    # 2. Full details CSV
    df.to_csv(output_dir / 'details_full.csv', index=False)
    print(f"✓ Saved: {output_dir / 'details_full.csv'}")

    # 3. Baseline analysis JSON
    baseline_analysis = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'dataset': dataset_name,
            'model': model_name,
            'source_file': result_file,
        },
        'summary_stats': {
            'overall_accuracy': float(df['effective_correct'].mean()),
            'total_examples': len(df),
            'num_tasks': df['task'].nunique(),
        },
        'by_size': size_analysis,
        'by_task_and_size': task_size_analysis,
        'failure_patterns': failure_patterns,
        'threshold_recommendations': threshold_recs,
    }

    with open(output_dir / 'baseline_analysis.json', 'w') as f:
        json.dump(baseline_analysis, f, indent=2)
    print(f"✓ Saved: {output_dir / 'baseline_analysis.json'}")

    # 4. Performance by size CSV
    if size_analysis:
        size_df = pd.DataFrame(size_analysis).T
        size_df.to_csv(output_dir / 'performance_by_size.csv')
        print(f"✓ Saved: {output_dir / 'performance_by_size.csv'}")

    # 5. Task × Size matrix CSV
    if task_size_analysis:
        # Flatten for CSV
        matrix_rows = []
        for task, sizes in task_size_analysis.items():
            row = {'task': task}
            for size, metrics in sizes.items():
                row[f'{size}_accuracy'] = metrics.get('accuracy', 0)
                row[f'{size}_count'] = metrics.get('count', 0)
            matrix_rows.append(row)

        matrix_df = pd.DataFrame(matrix_rows)
        matrix_df.to_csv(output_dir / 'task_size_matrix.csv', index=False)
        print(f"✓ Saved: {output_dir / 'task_size_matrix.csv'}")

    # 6. Performance by task CSV
    if 'task' in df.columns:
        task_df = df.groupby('task')['effective_correct'].agg(['mean', 'count']).reset_index()
        task_df.columns = ['task', 'accuracy', 'count']
        task_df.to_csv(output_dir / 'performance_by_task.csv', index=False)
        print(f"✓ Saved: {output_dir / 'performance_by_task.csv'}")

    # 7. Threshold recommendations TXT
    with open(output_dir / 'threshold_recommendations.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("THRESHOLD RECOMMENDATIONS FOR MCP-BASED APPROACH\n")
        f.write("=" * 80 + "\n\n")

        f.write("Cell Count Thresholds:\n")
        for key, value in threshold_recs.get('cell_count_thresholds', {}).items():
            f.write(f"  {key}: {value} cells\n")

        if threshold_recs.get('strategy_recommendations'):
            f.write("\nStrategy Recommendations:\n")
            for rec in threshold_recs['strategy_recommendations']:
                f.write(f"  • {rec}\n")

        if threshold_recs.get('key_insights'):
            f.write("\nKey Insights:\n")
            for insight in threshold_recs['key_insights']:
                f.write(f"  • {insight}\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"✓ Saved: {output_dir / 'threshold_recommendations.txt'}")

    return output_dir


def print_summary(df, size_analysis, failure_patterns, threshold_recs, dataset_name, model_name):
    """Print analysis summary to console."""
    print("\n" + "=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Total Examples: {len(df)}")
    print(f"Overall Accuracy: {df['effective_correct'].mean():.4f}")

    # By size
    if size_analysis:
        print("\n" + "-" * 80)
        print("PERFORMANCE BY TABLE SIZE")
        print("-" * 80)

        for size in ['small', 'medium', 'large']:
            if size in size_analysis:
                stats = size_analysis[size]
                print(f"\n{size.upper()}:")
                print(f"  Count: {stats['count']}")
                print(f"  Accuracy: {stats['accuracy']:.4f}")
                if 'avg_cells' in stats:
                    print(f"  Avg Cells: {stats['avg_cells']:.0f} " +
                          f"(range: {stats.get('min_cells', 0)}-{stats.get('max_cells', 0)})")
                if 'avg_rows' in stats and 'avg_cols' in stats:
                    print(f"  Avg Dimensions: {stats['avg_rows']:.1f} rows × {stats['avg_cols']:.1f} cols")

    # Worst tasks
    if failure_patterns.get('worst_performing_tasks'):
        print("\n" + "-" * 80)
        print("WORST PERFORMING TASKS")
        print("-" * 80)
        for task_info in failure_patterns['worst_performing_tasks'][:5]:
            print(f"  {task_info['task']:35s} - Accuracy: {task_info['accuracy']:.4f}")

    # Best tasks
    if failure_patterns.get('best_performing_tasks'):
        print("\n" + "-" * 80)
        print("BEST PERFORMING TASKS")
        print("-" * 80)
        for task_info in failure_patterns['best_performing_tasks'][:5]:
            print(f"  {task_info['task']:35s} - Accuracy: {task_info['accuracy']:.4f}")

    # Recommendations
    if threshold_recs.get('strategy_recommendations'):
        print("\n" + "-" * 80)
        print("MCP STRATEGY RECOMMENDATIONS")
        print("-" * 80)
        for rec in threshold_recs['strategy_recommendations']:
            print(f"  • {rec}")

    if threshold_recs.get('key_insights'):
        print("\n" + "-" * 80)
        print("KEY INSIGHTS FOR THESIS")
        print("-" * 80)
        for insight in threshold_recs['key_insights']:
            print(f"  • {insight}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python3 analysis/analyze_results.py <result.jsonl>")
        print("\nExample:")
        print("  python3 analysis/analyze_results.py inference_results/stratified_dataset.llama-3.2-3b.result.jsonl")
        sys.exit(1)

    result_file = sys.argv[1]

    if not os.path.exists(result_file):
        print(f"ERROR: File not found: {result_file}")
        sys.exit(1)

    # Extract names
    dataset_name, model_name = extract_dataset_and_model(result_file)

    # Step 1: Re-evaluate to get accurate scores
    df = reevaluate_results(result_file)

    if df is None:
        print("\nERROR: Re-evaluation failed")
        sys.exit(1)

    # Step 2: Analyze results
    print("\n" + "=" * 80)
    print("STEP 2: ANALYZING RESULTS")
    print("=" * 80)

    print("\nComputing by-size metrics...")
    size_analysis = compute_size_metrics(df)

    print("Computing task × size matrix...")
    task_size_analysis = analyze_by_task_and_size(df)

    print("Identifying failure patterns...")
    failure_patterns = identify_failure_patterns(df)

    print("Generating threshold recommendations...")
    threshold_recs = compute_threshold_recommendations(size_analysis, task_size_analysis)

    # Step 3: Save all results
    output_dir = save_all_results(
        df, size_analysis, task_size_analysis, failure_patterns,
        threshold_recs, dataset_name, model_name, result_file
    )

    # Step 4: Print summary
    print_summary(df, size_analysis, failure_patterns, threshold_recs, dataset_name, model_name)

    print("\n" + "=" * 80)
    print(f"✓ All results saved to: {output_dir}/")
    print("=" * 80)
    print("\nGenerated files:")
    print("  - accurate_by_size.json           # Precise by-size metrics")
    print("  - details_full.csv                # Full details with correct column")
    print("  - baseline_analysis.json          # Complete analysis with insights")
    print("  - performance_by_size.csv         # Size breakdown (ready for plots)")
    print("  - performance_by_task.csv         # Task breakdown")
    print("  - task_size_matrix.csv            # Task × Size matrix")
    print("  - threshold_recommendations.txt   # MCP strategy recommendations")

    return 0


if __name__ == "__main__":
    sys.exit(main())
