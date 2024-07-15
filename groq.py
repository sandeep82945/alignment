import json
from tqdm.auto import tqdm
import os
from typing import Dict, List
from groq import Groq

os.environ["GROQ_API_KEY"] = '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'

# LLAMA3_70B_INSTRUCT = "llama3-70b-8192"
LLAMA3_8B_INSTRUCT = "llama3-70b-8192"

DEFAULT_MODEL = LLAMA3_8B_INSTRUCT

client = Groq()

def assistant(content: str):
    return { "role": "assistant", "content": content }

def user(content: str):
    return { "role": "user", "content": content }

def chat_completion(
    messages: List[Dict],
    model = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    
    response = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        top_p=top_p,
    )
    return response.choices[0].message.content
        

def completion(
    prompt: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    top_p: float = 0.9,
) -> str:
    return chat_completion(
        [user(prompt)],
        model=model,
        temperature=temperature,
        top_p=top_p,
    )

a = []
def complete_and_print(prompt: str, model: str = DEFAULT_MODEL):
    # print(f'==============\n{prompt}\n==============')
    response = completion(prompt, model)
    return response


# response = complete_and_print("hi")
# print(response)


# with open('mix_reviews/mix_neur_ai_review_nosumm.json') as f:
#     review_data = json.load(f)

# ai_review_dict = {}
# a = 0
# for key, value in tqdm(review_data.items(), total=len(review_data)):
#     prompt = \
#     f'''
#         Review: {value}
#         Your task is to remove the Extraneous information from the review 
#         for instance remove 'Here is the revised review' or 'Main Review' anything which makes it look like a non-human review.
#         All the reviews and points should not be modified, just remove the extra tokens.
#    ''' 
#     response = complete_and_print(prompt)
#     ai_review_dict[key] = response
    

# print(f'length of ai_review_dict: {len(ai_review_dict)}')

# # save the dict
# # if args.dataset == 'neurips':
# # save_path = 'mix_reviews/mix_neur_ai_review_nosumm.json'
# # else:
# # save_path = 'mix_reviews/mix_iclr_ai_review_nosumm.json'

# # with open(save_path, 'w') as f:
# #     json.dump(ai_review_dict, f, indent=4)
    
# with open('mix_reviews/mix_neur_ai_review_nosumm_no_extra_token.json', 'w') as f:
#     json.dump(ai_review_dict, f, indent=4)
