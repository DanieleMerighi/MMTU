#!/usr/bin/env python3
"""
MMTU Benchmark Evaluation Script (Thesis-optimized)

This script evaluates model performance on the MMTU benchmark with:
1. JSON output for easy analysis and thesis work
2. Metrics grouped by table size (_size_category)
3. Non-destructive results (doesn't delete previous runs)
4. Organized output: results_thesis/{dataset_name}/{model_name}/
5. Detailed metadata preservation

Usage:
    python3 evaluate.py <result_file.jsonl> [--n_jobs N] [--debug] [--viz]

Examples:
    # Evaluate stratified dataset results
    python3 evaluate.py inference_results/stratified_dataset.llama-3.2-3b.result.jsonl --n_jobs 4

    # With debug output
    python3 evaluate.py inference_results/stratified_dataset.llama-3.2-3b.result.jsonl --debug --viz

Output structure:
    results_thesis/{dataset_name}/{model_name}/
    ├── summary.json                    # Overall statistics and MMTU score
    ├── metrics_by_task.json            # Per-task breakdown
    ├── metrics_by_size.json            # By table size (small/medium/large)
    ├── metrics_by_task_and_size.json   # Task × Size matrix
    ├── details.jsonl                   # Full details (JSONL)
    └── details.csv                     # Full details (CSV)
"""

import pandas as pd
import os
import argparse
import numpy as np
from datetime import datetime
import json
from collections import defaultdict
import time
import utils.utils as utils

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

# Summary metric for each task (acc or f1)
summary_metric = {
    "NL2SQL": "acc",
    "Table-QA": "acc",
    "Table-Fact-Verification": "acc",
    "Error-Detect": "f1",
    "Data-transform-pbe": "acc",
    "Entity-Matching": "acc",
    "Table-needle-in-a-haystack": "acc",
    "Table-Locate-by-Row-Col": "acc",
    "Schema-Matching": "f1",
    "Data-transform-reshape": "acc",
    "Data-Imputation": "acc",
    "List-to-table": "acc",
    "Formula-prediction-context": "acc",
    "Transform-by-output-target-schema": "acc",
    "Transform-by-input-output-table": "acc",
    "semantic-transform": "acc",
    "semantic-join": "f1",
    "header-value-matching": "acc",
    "Arithmetic-Relationship": "f1",
    "Functional-Dependency": "f1",
    "String-Relationship": "f1",
    "Cell-entity-annotation": "acc",
    "Column-type-annotation": "acc",
    "Columns-property-anotation": "acc",
    "equi-join-detect": "f1",
}

evaluator_dict = {
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
    """
    Extract dataset name and model name from result file path.

    Examples:
        stratified_dataset.llama-3.2-3b.result.jsonl -> ('stratified_dataset', 'llama-3.2-3b')
        mmtu.jsonl.gpt-4o.result.jsonl -> ('mmtu', 'gpt-4o')
    """
    basename = os.path.basename(result_file)

    # Remove .result.jsonl suffix
    if basename.endswith('.result.jsonl'):
        basename = basename[:-len('.result.jsonl')]

    # Split by dots
    parts = basename.split('.')

    if len(parts) >= 2:
        # First part is dataset name, rest is model name
        dataset_name = parts[0]
        model_name = '.'.join(parts[1:])
    else:
        dataset_name = 'unknown'
        model_name = parts[0] if parts else 'unknown'

    return dataset_name, model_name


def compute_metrics_by_size(detail_results_df, metric_name='correct'):
    """
    Compute metrics grouped by table size category.

    Args:
        detail_results_df: DataFrame with evaluation details
        metric_name: Column to aggregate (default: 'correct')

    Returns:
        dict with metrics per size category
    """
    if '_size_category' not in detail_results_df.columns:
        return {}

    size_metrics = {}
    for size_cat in ['small', 'medium', 'large']:
        size_df = detail_results_df[detail_results_df['_size_category'] == size_cat]
        if len(size_df) > 0:
            size_metrics[size_cat] = {
                'count': len(size_df),
                'accuracy': size_df[metric_name].mean() if metric_name in size_df.columns else None,
                'correct': size_df[metric_name].sum() if metric_name in size_df.columns else None,
            }

    return size_metrics


def compute_metrics_by_task_and_size(all_details_df):
    """
    Compute metrics matrix: task × size_category.

    Returns:
        dict with structure: {task: {size: metrics}}
    """
    if '_size_category' not in all_details_df.columns or 'task' not in all_details_df.columns:
        return {}

    task_size_metrics = {}

    for task in all_details_df['task'].unique():
        task_df = all_details_df[all_details_df['task'] == task]
        task_size_metrics[task] = {}

        for size_cat in ['small', 'medium', 'large']:
            size_df = task_df[task_df['_size_category'] == size_cat]
            if len(size_df) > 0 and 'correct' in size_df.columns:
                try:
                    # Only compute if correct is numeric
                    if pd.api.types.is_numeric_dtype(size_df['correct']):
                        task_size_metrics[task][size_cat] = {
                            'count': len(size_df),
                            'accuracy': float(size_df['correct'].mean()),
                            'correct': int(size_df['correct'].sum()),
                        }
                except (TypeError, ValueError):
                    # Skip if can't compute
                    pass

    return task_size_metrics


def save_json_safe(data, filepath):
    """Save data to JSON, handling numpy types."""
    def convert(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        return obj

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2, default=convert)
    print(f"✓ Saved: {filepath}")


def evaluate_thesis(result_file, n_jobs=-1, debug=False, viz=False):
    """
    Main evaluation function with thesis-optimized output.
    """
    print("=" * 80)
    print("THESIS EVALUATION - MMTU Benchmark")
    print("=" * 80)

    # Extract dataset and model names
    dataset_name, model_name = extract_dataset_and_model(result_file)
    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Result file: {result_file}\n")

    # Create output directory structure
    output_dir = os.path.join("results_thesis", dataset_name, model_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}\n")

    # Load predictions
    print("Loading predictions...")
    preds = pd.read_json(result_file, lines=True)
    print(f"Loaded {len(preds)} predictions\n")

    if preds.shape[0] == 0:
        print("ERROR: Empty predictions file")
        return

    # Ensure task column exists
    if "task" not in preds.columns:
        preds["task"] = preds["metadata"].apply(lambda x: json.loads(x)["task"])

    # Storage for results
    all_task_metrics = {}
    all_details = []
    overall_stats = {
        'dataset': dataset_name,
        'model': model_name,
        'timestamp': datetime.now().isoformat(),
        'total_examples': len(preds),
        'tasks': {},
    }

    # Evaluate each task
    print("Evaluating tasks...")
    print("-" * 80)

    for task, preds_group in preds.groupby("task"):
        print(f"\n[{task}] Processing {len(preds_group)} examples...")

        # Create debug/viz directories if needed
        debug_dir = os.path.join(output_dir, "debug", task) if debug else None
        viz_dir = os.path.join(output_dir, "viz", task) if viz else None
        if debug_dir:
            os.makedirs(debug_dir, exist_ok=True)
        if viz_dir:
            os.makedirs(viz_dir, exist_ok=True)

        # Run evaluator
        evaluator = evaluator_dict[task]
        avg_results, detail_results, error_cnt = evaluator.evaluate(
            preds_group,
            debug_dir=debug_dir,
            viz_dir=viz_dir,
            n_jobs=n_jobs
        )

        # Merge back size/dimension metadata from original preds
        # We need to merge on a common key - try using 'test_case' or index
        size_cols = ['_size_category', '_table_rows', '_table_cols', '_table_cells']
        available_size_cols = [col for col in size_cols if col in preds_group.columns]

        if available_size_cols:
            # Create a mapping from test_case (or other identifier) to size metadata
            if 'test_case' in preds_group.columns and 'test_case' in detail_results.columns:
                size_metadata = preds_group[['test_case'] + available_size_cols].drop_duplicates('test_case')
                detail_results = detail_results.merge(size_metadata, on='test_case', how='left')
            else:
                # Fallback: just add the columns with the first N values matching detail_results length
                preds_subset = preds_group.reset_index(drop=True).iloc[:len(detail_results)]
                for col in available_size_cols:
                    detail_results[col] = preds_subset[col].values

        # Get primary metric for this task
        primary_metric = summary_metric[task]
        if isinstance(primary_metric, list):
            primary_metric = primary_metric[0]

        # Compute overall accuracy for this task
        task_accuracy = avg_results[primary_metric].mean() if primary_metric in avg_results.columns else None

        # Compute metrics by size category (safely)
        size_metrics = {}
        try:
            if 'correct' in detail_results.columns and '_size_category' in detail_results.columns:
                # Check if 'correct' is numeric
                if pd.api.types.is_numeric_dtype(detail_results['correct']):
                    size_metrics = compute_metrics_by_size(detail_results, metric_name='correct')
        except Exception as e:
            print(f"  ⚠ Could not compute size metrics: {e}")

        # Store task metrics
        all_task_metrics[task] = {
            'count': len(preds_group),
            'metric_type': primary_metric,
            'accuracy': float(task_accuracy) if task_accuracy is not None and not pd.isna(task_accuracy) else None,
            'by_size': size_metrics,
            'by_dataset': avg_results.to_dict('records') if not avg_results.empty else [],
        }

        overall_stats['tasks'][task] = {
            'count': len(preds_group),
            'accuracy': float(task_accuracy) if task_accuracy is not None and not pd.isna(task_accuracy) else None,
        }

        # Store all details
        detail_results['task'] = task
        all_details.append(detail_results)

        print(f"  ✓ Accuracy: {task_accuracy:.4f}" if task_accuracy is not None else "  ✓ Completed")
        if size_metrics:
            print(f"  ✓ By size: ", end="")
            for size, metrics in size_metrics.items():
                acc = metrics.get('accuracy')
                if acc is not None:
                    print(f"{size}={acc:.3f} ", end="")
            print()

    # Combine all details
    print("\n" + "-" * 80)
    print("Computing aggregate metrics...")
    all_details_df = pd.concat(all_details, axis=0, ignore_index=True)

    # Compute task × size matrix
    task_size_matrix = compute_metrics_by_task_and_size(all_details_df)

    # Compute overall metrics by size
    overall_by_size = {}
    if '_size_category' in all_details_df.columns and 'correct' in all_details_df.columns:
        try:
            # Only compute if correct is numeric
            if pd.api.types.is_numeric_dtype(all_details_df['correct']):
                for size_cat in ['small', 'medium', 'large']:
                    size_df = all_details_df[all_details_df['_size_category'] == size_cat]
                    if len(size_df) > 0:
                        overall_by_size[size_cat] = {
                            'count': len(size_df),
                            'accuracy': float(size_df['correct'].mean()),
                            'correct': int(size_df['correct'].sum()),
                        }
        except (TypeError, ValueError) as e:
            print(f"⚠ Could not compute overall size metrics: {e}")

    # Compute overall MMTU score
    task_scores = [v['accuracy'] for v in all_task_metrics.values() if v['accuracy'] is not None]
    overall_mmtu_score = np.mean(task_scores) if task_scores else None

    overall_stats['mmtu_score'] = float(overall_mmtu_score) if overall_mmtu_score is not None else None
    overall_stats['by_size'] = overall_by_size

    # Save results
    print("\nSaving results...")
    print("-" * 80)

    # 1. Summary JSON
    save_json_safe(overall_stats, os.path.join(output_dir, "summary.json"))

    # 2. Metrics by task JSON
    save_json_safe(all_task_metrics, os.path.join(output_dir, "metrics_by_task.json"))

    # 3. Metrics by size JSON
    save_json_safe(overall_by_size, os.path.join(output_dir, "metrics_by_size.json"))

    # 4. Task × Size matrix JSON
    save_json_safe(task_size_matrix, os.path.join(output_dir, "metrics_by_task_and_size.json"))

    # 5. Full details JSON (may be large)
    print(f"Saving detailed results ({len(all_details_df)} rows)...")
    all_details_df.to_json(
        os.path.join(output_dir, "details.jsonl"),
        orient='records',
        lines=True
    )
    print(f"✓ Saved: {os.path.join(output_dir, 'details.jsonl')}")

    # 6. Also save CSV for compatibility
    all_details_df.to_csv(os.path.join(output_dir, "details.csv"), index=False)
    print(f"✓ Saved: {os.path.join(output_dir, 'details.csv')}")

    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"\nOverall MMTU Score: {overall_mmtu_score:.4f}" if overall_mmtu_score is not None else "\nOverall MMTU Score: N/A")

    if overall_by_size:
        print("\nAccuracy by Table Size:")
        for size, metrics in overall_by_size.items():
            print(f"  {size.capitalize():8s}: {metrics['accuracy']:.4f} ({metrics['correct']}/{metrics['count']})")

    print(f"\nResults saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  - summary.json                    # Overall statistics")
    print("  - metrics_by_task.json            # Per-task breakdown")
    print("  - metrics_by_size.json            # By table size (small/medium/large)")
    print("  - metrics_by_task_and_size.json   # Task × Size matrix")
    print("  - details.jsonl                   # Full details (JSONL)")
    print("  - details.csv                     # Full details (CSV)")

    return overall_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Thesis-optimized evaluation for MMTU benchmark"
    )
    parser.add_argument("result_file", type=str, help="Path to .result.jsonl file")
    parser.add_argument("--n_jobs", default=-1, type=int, help="Number of parallel jobs (-1 for all CPUs)")
    parser.add_argument("--debug", action="store_true", default=False, help="Save debug files")
    parser.add_argument("--viz", action="store_true", default=False, help="Save visualization files")
    args = parser.parse_args()

    if args.n_jobs < 0:
        args.n_jobs = os.cpu_count()

    assert os.path.exists(args.result_file), f"Result file {args.result_file} does not exist"

    # Run evaluation
    evaluate_thesis(
        args.result_file,
        n_jobs=args.n_jobs,
        debug=args.debug,
        viz=args.viz
    )
