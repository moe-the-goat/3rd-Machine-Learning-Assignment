"""
Main Pipeline: Complete Machine Learning Assignment
Orchestrates all tasks: EDA, Baseline, Model Training, and Error Analysis.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))

def print_header(task_name):
    """Print a formatted task header."""
    print("\n")
    print("=" * 80)
    print(f"  {task_name}")
    print("=" * 80)
    print()


def main():
    """Run the complete ML pipeline."""
    print("=" * 80)
    print("  MACHINE LEARNING ASSIGNMENT - COMPLETE PIPELINE")
    print("  Travel Destination Country Classification")
    print("=" * 80)
    print("\nExecuting all tasks. Output will be saved to separate directories.")
    
    # Task 1
    print_header("TASK 1: LEARNING TASK DEFINITION")
    try:
        from task1_learning_task import define_learning_task
        define_learning_task()
        print("✅ Task 1 complete!")
    except Exception as e:
        print(f"❌ Task 1 failed: {e}")
    
    # Task 2
    print_header("TASK 2: EXPLORATORY DATA ANALYSIS")
    try:
        from task2_eda import run_eda
        run_eda()
        print("✅ Task 2 complete!")
    except Exception as e:
        print(f"❌ Task 2 failed: {e}")
    
    # Task 3
    print_header("TASK 3: BASELINE MODEL (k-NN)")
    try:
        from task3_baseline import run_baseline
        run_baseline()
        print("✅ Task 3 complete!")
    except Exception as e:
        print(f"❌ Task 3 failed: {e}")
    
    # Task 4
    print_header("TASK 4: PROPOSED MACHINE LEARNING MODELS")
    try:
        from task4_models import run_models
        run_models()
        print("✅ Task 4 complete!")
    except Exception as e:
        print(f"❌ Task 4 failed: {e}")
    
    # Task 5
    print_header("TASK 5: ERROR ANALYSIS")
    try:
        from task5_error_analysis import run_error_analysis
        run_error_analysis()
        print("✅ Task 5 complete!")
    except Exception as e:
        print(f"❌ Task 5 failed: {e}")
    
    # Summary
    print("\n")
    print("=" * 80)
    print("  PIPELINE COMPLETE")
    print("=" * 80)
    print("\nOutput directories:")
    
    base_dir = SCRIPT_DIR.parent
    output_dirs = [
        "Task1_Output",
        "EDA_Output", 
        "Task3_Baseline",
        "Task4_Models",
        "Task5_ErrorAnalysis"
    ]
    
    for dir_name in output_dirs:
        dir_path = base_dir / dir_name
        if dir_path.exists():
            file_count = len(list(dir_path.glob("*")))
            print(f"  ✓ {dir_name}/ ({file_count} files)")
        else:
            print(f"  ✗ {dir_name}/ (not created)")
    
    print("\nCheck each directory for detailed results.")
    print("=" * 80)

if __name__ == "__main__":
    main()
