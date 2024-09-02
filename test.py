import json
with open('data/final_input_output_pairs.json') as f:
    dic = json.load(f)

from grok import complete_and_print_llama



def evaluate_pairwise(prediction, ground_truth):
    """
    Maps pairwise predictions to win/loss/tie based on human annotations.

    Args:
    - prediction (str): The model's prediction ("A won", "B won", or "tie").
    - ground_truth (str): The human annotation ("A won", "B won", or "tie").

    Returns:
    - result (str): "win", "loss", or "tie".
    """
    if prediction == ground_truth:
        if prediction == "C":
            return "C"
        else:
            return "win"
    elif prediction == "C" or ground_truth == "C":
        return "tie"
    else:
        return "loss"

def calculate_metrics(predictions, ground_truths):
    """
    Calculate the win rate, loss rate, and tie rate.

    Args:
    - predictions (list of str): Model's predictions ("A won", "B won", or "tie").
    - ground_truths (list of str): Human annotations ("A won", "B won", or "tie").

    Returns:
    - metrics (dict): Dictionary containing 'win_rate', 'loss_rate', and 'tie_rate'.
    """
    assert len(predictions) == len(ground_truths), "Predictions and ground truths must be of the same length."

    total = len(predictions)
    win_count = 0
    loss_count = 0
    tie_count = 0

    for pred, gt in zip(predictions, ground_truths):
        result = evaluate_pairwise(pred, gt)
        if result == "win":
            win_count += 1
        elif result == "loss":
            loss_count += 1
        elif result == "tie":
            tie_count += 1

    metrics = {
        "win_rate": win_count / total,
        "loss_rate": loss_count / total,
        "tie_rate": tie_count / total
    }

    return metrics

import re

import re

def extract_winner(text):
    """
    Extracts the winning system (A or B) from the given text.

    Args:
    - text (str): A string containing the comparison result.

    Returns:
    - winner (str): The winning system ("A" or "B").
    """
    # Generic regex pattern to capture verdict enclosed in brackets
    pattern = r'\[\[?([ABC])\]?\]'

    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    
    if match:
        return match.group(1)
    
    return None

    

progress_file = 'results_consistency.json'
import os
# Initialize or load previous progress
if os.path.exists(progress_file):
    with open(progress_file, 'r') as f:
        data = json.load(f)
        pred = data.get('pred', [])
        true = data.get('true', [])
        processed_inputs = data.get('processed_inputs', [])
else:
    pred = []
    true = []
    processed_inputs = []

from tqdm import tqdm

def call_llm(input):
    return complete_and_print_llama(input)

# for item in tqdm(dic['test']['fluency']):
#     input_text = item['input']
#     output_text = item['output']
    
#     # Skip if this input has already been processed
#     if input_text in processed_inputs:
#         continue

#     # Call LLM and extract predictions
#     response = call_llm(input_text)
#     extracted_pred = extract_winner(response[-100:])
#     extracted_true = extract_winner(output_text)
    
#     if extracted_pred is None or extracted_true is None:
#         continue

#     # Append to lists
#     true.append(extracted_true)
#     pred.append(extracted_pred)
#     processed_inputs.append(input_text)

#     # Save progress after each item
#     with open(progress_file, 'w') as f:
#         json.dump({
#             'pred': pred,
#             'true': true,
#             'processed_inputs': processed_inputs
#         }, f, indent=4)

#     # Optionally, print progress
#     print("Processed:", input_text)
#     print("True labels:", true)
#     print("Predicted labels:", pred)
#     print("--------------------")

metrics = calculate_metrics(pred, true)
print(metrics)