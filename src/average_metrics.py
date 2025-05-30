import os
import re
from collections import defaultdict

def average_metrics_from_file(file_path):
    metrics = defaultdict(list)

    with open(file_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                # Join all parts between index 2 and the last (excluding value)
                metric = ' '.join(parts[2:-1])
                value = parts[-1]
                try:
                    metrics[metric].append(float(value))
                except ValueError:
                    continue  # skip lines where value is not a float

    return {metric: sum(vals) / len(vals) for metric, vals in metrics.items()}


def process_all_results(folder_path):
    results = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            match = re.search(r"(\d+)\.txt$", filename)
            if match:
                dataset_name = filename.split('_')[2]
                possible_name = filename.split('_')[3]
                if possible_name != "v2":
                    dataset_name = dataset_name + "_" + possible_name
                k = int(match.group(1))
                file_path = os.path.join(folder_path, filename)
                averages = average_metrics_from_file(file_path)
                results.append((dataset_name, k, averages))

    # Sort by k
    results.sort(key=lambda x: x[0])
    return results

# Example usage
folder_path = "./results_gmms_seeds"
# folder_path = "./resultskm"
all_results = process_all_results(folder_path)
print(all_results)
for dataset_name, k, metrics in all_results:
    print("Dataset:", dataset_name)
    print(f"k = {k}: ", end='')
    print(", ".join(f"{metric}: {value:.4f}" for metric, value in metrics.items()))
