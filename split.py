import json
import random

# Load JSON data from a file
with open('summeval.json', 'r') as file:
    data = json.load(file)

# Group by 'doc_id'
from collections import defaultdict
grouped_data = defaultdict(list)
for item in data:
    grouped_data[item["doc_id"]].append(item)

# Create a list of unique doc_ids and shuffle it for random splitting
doc_ids = list(grouped_data.keys())
random.shuffle(doc_ids)

# Split the doc_ids into train and test
split_idx = int(len(doc_ids) * 0.8)  # 80% for training
train_doc_ids = set(doc_ids[:split_idx])
test_doc_ids = set(doc_ids[split_idx:])

# Allocate records to train and test based on their doc_id
train_data = [item for doc_id in train_doc_ids for item in grouped_data[doc_id]]
test_data = [item for doc_id in test_doc_ids for item in grouped_data[doc_id]]

# Save the split data to JSON files
with open('train_data.json', 'w') as f:
    json.dump(train_data, f, indent=4)

with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=4)

print(f"Saved {len(train_data)} records to train_data.json and {len(test_data)} records to test_data.json.")
