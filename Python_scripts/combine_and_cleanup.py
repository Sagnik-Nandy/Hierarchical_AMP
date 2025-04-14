import pandas as pd
import os
import glob

print("Combining partial result files...", flush=True)

n_values = [300, 400]
combined = {}

for n in n_values:
    path = f"Results/partial_result_{n}.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path, index_col=0)
            print(f"Loaded partial_result_{n}.csv with columns: {df.columns.tolist()}", flush=True)
            if 'Metric' in df.columns:
                combined[n] = df.set_index("Metric").T.to_dict('records')[0]
            else:
                print(f"Error: 'Metric' column missing in partial_result_{n}.csv", flush=True)
        except Exception as e:
            print(f"Error processing partial_result_{n}.csv: {e}", flush=True)
    else:
        print(f"Missing: partial_result_{n}.csv", flush=True)

# Save combined results
final_df = pd.DataFrame(combined).T
final_df.to_csv("Results/chat_coeff_n.csv")
print("Saved combined results to Results/chat_coeff_n.csv", flush=True)

# Remove intermediate files
partial_files = glob.glob("Results/partial_result_*.csv")
for f in partial_files:
    os.remove(f)
    print(f"Removed {f}", flush=True)

print("Cleanup complete.", flush=True)