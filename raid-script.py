import argparse
import json
from mosaic import Mosaic
from datasets import load_dataset

model_list = ["TowerBase-7B-v0.1", "TowerBase-13B-v0.1", "Llama-2-7b-chat-hf", "Llama-2-7b-hf"]

def process_texts(dataset, mosaic, output_file):
    results = []
    i = 0

    # Process each item and store the results in a list
    for item in dataset:
        i += 1
        text = item["generation"]
        id = item["id"]

        # Compute scores using the mosaic object
        avg_score, max_score, min_score = mosaic.compute_end_scores(text)

        # Use the average score as the final score (or another approach you prefer)
        final_score = avg_score  # You can change this logic, avg is the default one

        # Append the result as a dictionary with 'id' and 'score'
        result = {
            "id": id,
            "score": final_score
        }
        results.append(result)

        # Optionally print progress
        if i % 1000 == 0:
            print(f"{i} entries done")

    # Step 2: Write the entire results list to the output file
    with open(output_file, "w") as f:
        json.dump(results, f)  # Write the complete list at once in JSON format

    print(f"Processing complete. Results saved to {output_file}")

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process texts and compute scores using Mosaic models.")
    parser.add_argument('--output', type=str, required=True, help='Path to the output JSON file')

    args = parser.parse_args()

    # Assign model paths (ensure we strip leading/trailing spaces from model names)
    model_list = [model_name.strip() for model_name in model_list]
    
    # Initialize the Mosaic object with the provided model list
    mosaic = Mosaic(model_list)

    # Load the dataset
    dataset = load_dataset("liamdugan/raid", "raid")

    # Process texts and save to output file
    process_texts(dataset, mosaic, args.output)