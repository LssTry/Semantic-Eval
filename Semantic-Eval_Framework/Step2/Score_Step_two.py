import os
import time
import pandas as pd
from transformers import pipeline

input_excel_path = r"data/examples.xlsx"#input file
output_csv_path = r"examples_step2.csv"#output file

classifier = pipeline("text-classification", model="roberta-large-mnli")
df = pd.read_excel(input_excel_path, usecols=["reference", "candidate"])
total_rows = len(df)

if os.path.exists(output_csv_path):
    processed_df = pd.read_csv(output_csv_path)
    processed_rows = len(processed_df)
else:
    processed_rows = 0
    with open(output_csv_path, 'w', encoding='utf-8') as f:
        f.write("Confidence_Score\n")

remaining_rows = total_rows - processed_rows
if remaining_rows <= 0:
    print("All data has been processed.")
    exit()
num_batches = 50
batch_size = remaining_rows // num_batches
if remaining_rows % num_batches != 0:
    batch_size += 1

for batch_idx in range(num_batches):
    start = processed_rows + batch_idx * batch_size
    end = min(start + batch_size, total_rows)

    if start >= end:
        break

    print(f"Processing batch {batch_idx + 1}: Rows {start + 1} to {end}")

    for row_idx in range(start, end):
        sentence1 = df.at[row_idx, "candidate"]
        sentence2 = df.at[row_idx, "reference"]

        try:
            result = classifier(f"{sentence1} </s> {sentence2}")[0]
            label = result['label']
            score = round(1 - result['score'], 4) if label.upper() == 'CONTRADICTION' else round(result['score'], 4) if label.upper() == 'NEUTRAL' else 1
            print("score", score)
            with open(output_csv_path, 'a', encoding='utf-8') as f:
                f.write(f"{label},{score}\n")

        except Exception as e:
            print(f"Error processing row {row_idx + 1}: {str(e)}")

        time.sleep(0.2)

    print(f"Batch {batch_idx + 1} completed.")

print("All data processing completed.")