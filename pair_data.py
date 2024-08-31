import json
from itertools import combinations
import pandas as pd
import yaml
import random
random_seed = 42
random.seed(random_seed)
import math
class SFTTrainingDataset:
    def __init__(self, input_file, split_ratio = 0.8):
        self.input_file = input_file
        self.data = self.load_json()
        #{'doc_id': 'dm-test-8bc8f134bc3ceaeec0df8f29e96e13d319717ab8', 'system_id': 'M17', 'source':
        self.doc_dict = self.group_by_doc_id()
        keys = list( self.doc_dict.keys())
        random.shuffle(keys)
        # Specify the split ratio (e.g., 80% for training, 20% for testing)

        # Calculate the split index
        split_index = math.floor(len(keys) * split_ratio)

        # Split the keys into training and testing sets
        train_keys = keys[:split_index]
        test_keys = keys[split_index:]
        self.train_dict = {key: self.doc_dict[key] for key in train_keys}
        self.test_dict = {key: self.doc_dict[key] for key in test_keys}
        print("# of training doc_ids = ", len(self.train_dict.keys()))
        print("# of testing doc_ids = ", len(self.test_dict.keys()))


    def load_json(self):
        with open(self.input_file, 'r') as file:
            return json.load(file)

    def group_by_doc_id(self):
        doc_dict = {}
        for entry in self.data:
            doc_id = entry['doc_id']
            if doc_id not in doc_dict:
                doc_dict[doc_id] = []
            doc_dict[doc_id].append(entry)
        return doc_dict

    @staticmethod
    def compare_scores(scores_a, scores_b):
        aspects = ['coherence', 'consistency', 'fluency', 'relevance', 'overall']
        results = {}
        for aspect in aspects:
            if scores_a[aspect] > scores_b[aspect]:
                results[aspect] = ' System A is better than B ; Answer is: [[A]]'
            elif scores_a[aspect] < scores_b[aspect]:
                results[aspect] = 'System B is better than A ; Answer is: [[B]]'
            else:
                results[aspect] = 'Its a tie between System A and B ; Answer is: [[C]]'
        return results

    def generate_training_data(self):
        training_data = []
        for doc_id, outputs in self.train_dict.items():
            pairs = combinations(outputs, 2)
            for a, b in pairs:
                comparison = self.compare_scores(a['scores'], b['scores'])
                training_data.append({
                    'doc_id': doc_id,
                    'source': a['source'],
                    'reference': a['reference'],
                    'system_id_a': a['system_id'],
                    'system_output_a': a['system_output'],
                    'scores_a': a['scores'],
                    'system_id_b': b['system_id'],
                    'system_output_b': b['system_output'],
                    'scores_b': b['scores'],
                    'comparison': comparison
                })
        return training_data
    
    def generate_testing_data(self):
        testing_data = []
        for doc_id, outputs in self.test_dict.items():
            pairs = combinations(outputs, 2)
            for a, b in pairs:
                comparison = self.compare_scores(a['scores'], b['scores'])
                testing_data.append({
                    'doc_id': doc_id,
                    'source': a['source'],
                    'reference': a['reference'],
                    'system_id_a': a['system_id'],
                    'system_output_a': a['system_output'],
                    'scores_a': a['scores'],
                    'system_id_b': b['system_id'],
                    'system_output_b': b['system_output'],
                    'scores_b': b['scores'],
                    'comparison': comparison
                })
        return testing_data

    def save_to_json(self, data, output_file):
        df = pd.DataFrame(data)
        pd.set_option('display.max_colwidth', None)  # Ensure the full content of each column is displayed
        print(df.head())
        df.to_json(output_file, orient='records', lines=True)

    def create_dataset(self, output_file=None, return_json=False):
        training_data = self.generate_training_data()
        testing_data = self.generate_testing_data()
        # if output_file:
        #     self.save_to_json(training_data, output_file)
        
        if return_json:
            return training_data, testing_data

input_file = 'train_data.json'
output_file = 'training_data_with_scores.json'
pair_output_file = 'final_input_output_pairs.json'

sft_dataset = SFTTrainingDataset(input_file)
paired_data_with_comparison_training_set, paired_data_with_comparison_testing_set = sft_dataset.create_dataset(output_file=output_file, return_json=True)


print("Training Set length:", len(paired_data_with_comparison_training_set))
print("Testing Set length:", len(paired_data_with_comparison_testing_set))
#Above list contains each: dict_keys(['doc_id', 'source', 'reference', 'system_id_a', 'system_output_a', 'scores_a', 'system_id_b', 'system_output_b', 'scores_b', 'comparison'], comparison -> {'coherence': 'B is better than A [[B]]', 'consisten...)

#---------------------------------Now making the input output--------------------------
class InputOutputPairs:
    def __init__(self, config):
        prompt_coh = open(config["Prompt_dir"]["coh"]).read()
        prompt_con = open(config["Prompt_dir"]["con"]).read()
        prompt_flu = open(config["Prompt_dir"]["flu"]).read()
        prompt_rel = open(config["Prompt_dir"]["rel"]).read()
        self.prompt_dict = {'coherence':prompt_coh, 'consistency':prompt_con, 'fluency':prompt_flu, 'relevance':prompt_rel}
        self.coherence_pairs = []
        self.consistency_pairs = []
        self.fluency_pairs = []
        self.relevance_pairs = []

    def create_input_output_pairs(self, training_data):
        for entry in training_data:
            for aspect, response in entry['comparison'].items():
                if aspect == 'overall':
                    continue
                prompt = self.prompt_dict[aspect]
                aspect_variable = getattr(self, f'{aspect}_pairs')
                aspect_variable.append({
                'input': f"{prompt}\n \n Source: {entry['source']}\n \n Reference: {entry['reference']}\n \n System A: {entry['system_output_a']}\n\n System B: {entry['system_output_b']}",
                'output': response
            })
                
        return {'coherence':self.coherence_pairs, 'consistency':self.consistency_pairs, 'fluency':self.fluency_pairs, 'relevance':self.relevance_pairs}


    @staticmethod
    def save_input_output_pairs_to_json(input_output_pairs, output_file):
        with open(output_file, 'w') as file:
            for pair in input_output_pairs:
                file.write(json.dumps(pair) + '\n')

# Load configuration
with open('../config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# Example usage


io_pairs = InputOutputPairs(config)
input_output_pairs_training = io_pairs.create_input_output_pairs(paired_data_with_comparison_training_set)
input_output_pairs_testing = io_pairs.create_input_output_pairs(paired_data_with_comparison_testing_set)

# if pair_output_file:
#     InputOutputPairs.save_input_output_pairs_to_json(input_output_pairs, pair_output_file)

result_json = {
    'train': input_output_pairs_training,
    'test': input_output_pairs_testing
}

with open(pair_output_file, 'w') as json_file:
    json.dump(result_json, json_file, indent=4)

print("JSON file saved successfully!")

# print(result_json['input_output_pairs'][0])

