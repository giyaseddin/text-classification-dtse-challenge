import pandas as pd
import json
from pathlib import Path

# visualize(file) -> reporting (first parent) -> src -> root
PROJECT_ROOT_PATH: Path = Path(__file__).parent.parent.parent


def compile_experiment_reports(base_dir=PROJECT_ROOT_PATH / 'reports'):
    experiment_folders = [f for f in base_dir.iterdir() if f.is_dir()]
    data = []

    # Iterate over each experiment directory
    for folder in experiment_folders:
        exp_data = {'Experiment': folder.name}
        report_paths = {
            'Overall': folder / 'clf_report.json',
            'Cyber_Label': folder / 'clf_report_cyber_label.json',
            'Environmental_Issue': folder / 'clf_report_environmental_issue.json'
        }

        # Extract macro avg f1-score from each report
        for key, path in report_paths.items():
            if path.exists():
                with open(path, 'r') as file:
                    report = json.load(file)
                    exp_data[f'{key}_F1_Score'] = round(report['macro avg']['f1-score'], 3)
            else:
                exp_data[f'{key}_F1_Score'] = None

        data.append(exp_data)

    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.sort_values(by='Experiment')  # Sorting by experiment name for better readability
    csv_path = base_dir / 'experiment_comparison_table.csv'
    df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to {csv_path}")
    print(df.to_markdown(index=False, floatfmt=".3f"))


def compile_experiment_reports_per_label(base_dir=PROJECT_ROOT_PATH / 'reports'):
    experiment_folders = [f for f in base_dir.iterdir() if f.is_dir()]
    data = {}

    # Iterate over each experiment directory
    for folder in experiment_folders:
        # Find all report JSON files in the folder

        for label in ["cyber_label", "environmental_issue"]:
            report_file = folder / f'clf_report_{label}.json'
            if label not in data:
                data[label] = []  # Initialize a list for this label if it doesn't exist

            if report_file.exists():
                with open(report_file, 'r') as file:
                    report = json.load(file)
                exp_data = {
                    'Experiment': folder.name,
                    'Precision': round(report['macro avg']['precision'], 3),
                    'Recall': round(report['macro avg']['recall'], 3),
                    'F1_Score': round(report['macro avg']['f1-score'], 3)
                }
            else:
                exp_data = {'Experiment': folder.name, 'Precision': None, 'Recall': None, 'F1_Score': None}

            data[label].append(exp_data)

    # Create DataFrames and save them
    for label, records in data.items():
        df = pd.DataFrame(records)
        df = df.sort_values(by='Experiment')  # Sorting by experiment name
        csv_path = base_dir / f'experiment_comparison_{label}.csv'
        df.to_csv(csv_path, index=False)
        print(f"{label.capitalize()} comparison table saved to {csv_path}")
        # Optional: Print the table in Markdown format for quick view
        print(f"{label.capitalize()} Comparison Table:")
        print(df.to_markdown(index=False, floatfmt=".3f"))


# Run the script
compile_experiment_reports()
compile_experiment_reports_per_label()
