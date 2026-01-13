"""
Main Pipeline: Country Classification from Travel Descriptions
Runs all tasks: EDA, Baseline, Models (RF + Transformer), and Error Analysis.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))


def print_header(task_name):
    print("\n" + "=" * 70)
    print(f"  {task_name}")
    print("=" * 70 + "\n")


def main():
    print("=" * 70)
    print("  ML ASSIGNMENT: TRAVEL DESTINATION CLASSIFICATION")
    print("=" * 70)
    
    # Task 1: Learning Task Definition
    print_header("TASK 1: LEARNING TASK DEFINITION")
    try:
        from task1_learning_task import define_learning_task
        define_learning_task()
        print("Done!")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Task 2: EDA
    print_header("TASK 2: EXPLORATORY DATA ANALYSIS")
    try:
        from task2_eda import run_eda
        run_eda()
        print("Done!")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Task 3: Baseline (KNN)
    print_header("TASK 3: BASELINE MODEL (k-NN)")
    try:
        from task3_baseline import run_baseline
        run_baseline()
        print("Done!")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Task 4: Main Models (Random Forest + Transformer)
    print_header("TASK 4: MAIN MODELS (Random Forest + Transformer)")
    try:
        from task4_models import run_models
        run_models()
        print("Done!")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Task 5: Error Analysis
    print_header("TASK 5: ERROR ANALYSIS")
    try:
        from task5_error_analysis import run_error_analysis
        run_error_analysis()
        print("Done!")
    except Exception as e:
        print(f"Failed: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("  PIPELINE COMPLETE")
    print("=" * 70)
    
    base_dir = SCRIPT_DIR.parent
    results_dir = base_dir / "Results"
    output_dirs = ["Task1_Output", "Task2_EDA", "Task3_Baseline", 
                   "Task4_Models", "Task5_ErrorAnalysis"]
    
    print("\nOutput folders in Results/:")
    for name in output_dirs:
        path = results_dir / name
        if path.exists():
            count = len(list(path.glob("*")))
            print(f"  Results/{name}/ ({count} files)")
    
    print("\nOther_Models_Tried/ contains SVM, TextCNN, LogReg experiments")
    print("=" * 70)


if __name__ == "__main__":
    main()
