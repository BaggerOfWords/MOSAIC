from mosaic import Mosaic
from datasets import load_dataset

model_list = ["TowerBase-7B-v0.1", "TowerBase-13B-v0.1", "Llama-2-7b-chat-hf", "Llama-2-7b-hf"]

text = "This is an example sentence"

mosaic = Mosaic(model_list)

# Compute scores using the mosaic object
avg_score, max_score, min_score = mosaic.compute_end_scores(text)

# Use the average score as the final score (or another approach you prefer)
final_score = avg_score  # You can change this logic, avg is the default one