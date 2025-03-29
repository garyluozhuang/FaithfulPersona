import multiprocessing
import argparse
from datasets import load_dataset
import pandas as pd
import json
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from tqdm import tqdm
import time


code_temperature = 0.2
code_top_p = 0.1
text_temperature = 0.7
text_top_p = 0.8
model = "gpt-3.5-turbo"
# model
# Please make sure to set the OpenAI API key in your environment variables or replace the empty string with your actual API key.
code_model = ChatOpenAI(model=model, 
    openai_api_base="https://admin.openai.one/v1", 
    api_key="",  
    temperature=code_temperature,
    model_kwargs={"top_p": code_top_p})
text_model = ChatOpenAI(model=model,
    openai_api_base="https://admin.openai.one/v1", 
    api_key="", 
    temperature=text_temperature,
    model_kwargs={"top_p": text_top_p})

# Params
def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--group', type=str, default='valid', help='Group name (default: valid)')
    parser.add_argument('--sample_num', type=int, default=1, help='Sample number (default: 1)')
    parser.add_argument('--user_id', type=int, default=179736, help='User ID (default: 179736)')
    parser.add_argument('--method', type=str, default='base_personalization', help='Method (default: base_personalization)')
    parser.add_argument('--iter_count', type=int, default=1, help='Iteration count (default: 1)')
    return parser.parse_args()


args = parse_arguments()
group = args.group
sample_num = args.sample_num
user_id = args.user_id
method = args.method
iter_count = args.iter_count

folder_path_a = f"./result/explanation/{group}/exp_disco_personalization/{user_id}/iter={iter_count}/sample={sample_num}/"
if method == "base_personalization":
    folder_path_b = f"./result/explanation/{group}/base_personalization/{user_id}/sample={sample_num}/"
elif method == "base_personalization_consistency":
    folder_path_b = f"./result/explanation/{group}/base_personalization_consistency/{user_id}/sample={sample_num}/"
elif method == "disco":
    folder_path_b = f"./result/explanation/{group}/exp_disco/iter={iter_count}/sample={sample_num}/"

with open(f'user_profile/user_{user_id}_ground_truth_tag.txt', 'r') as file:
    user_inquiry_history = file.read()

data = load_dataset("deepmind/code_contests", split=group)
df = pd.DataFrame(data)
target_idx_list = [
    idx for idx, value in df.iterrows() if "<image>" not in value["description"] and
    (3 in value["solutions"]["language"] or 1 in value["solutions"]["language"])
]

def create_prompt(name, problem, code, explanation_a, explanation_b, user_inquiry_history, reverse=False):
    if reverse:
        explanation_a, explanation_b = explanation_b, explanation_a

    prompt = f"""Imagine you are a programmer with the following inquiry history:
<user_inquiry_history>
{user_inquiry_history}
</user_inquiry_history>

Your are given a code contest problem and an accepted correct solution code:
<problem_name>{name}</problem_name>
<problem_description>{problem}</problem_description>
<code>{code}</code>

You are trying to understand the code. You have two code explanations to choose from:
[
    {{
        "method": "method_0",
        "explanation": {explanation_a}
    }},
    {{
        "method": "method_1",
        "explanation": {explanation_b}
    }}
]

Now create a leaderboard by ranking the two code explanations based on your skill level, background and preferences inferred from the content in the inquiry history, to determine which explanation is more helpful and informative to you.

Please provide your decision and why in the following JSON format:
{{
    "leaderboard": [], // A sorted list of the methods from best to worst, e.g., ["method_0", "method_1"]
    "reasoning": ""
}}"""
    
    return prompt

def get_explanations_and_decisions_with_retry(idx):
    def get_explanations_and_decisions():
        df_sample = df.loc[idx, :]
        problem = df_sample["description"]
        name = df_sample["name"]
        # if name == "1601_A. Array Elimination":

        solutions = (soln for lang, soln in zip(df_sample["solutions"]["language"], df_sample["solutions"]["solution"]) if lang in (1, 3))
        code = min(solutions, key=len)

        with open(folder_path_a + name + "/best_explanation.txt", "r") as file:
            explanation_a = file.read()
            explanation_a = json.loads(explanation_a)
            if "personalized_explanation" in explanation_a:
                explanation_a = {"personalized_explanation": explanation_a["personalized_explanation"]}
                explanation_a = json.dumps(explanation_a)
            else:
                explanation_a = json.dumps(explanation_a)

            

        with open(folder_path_b + name + "/best_explanation.txt", "r") as file:
            explanation_b = file.read()
            explanation_b = json.loads(explanation_b)
            if "personalized_explanation" in explanation_b:
                explanation_b = {"personalized_explanation": explanation_b["personalized_explanation"]}
                explanation_b = json.dumps(explanation_b)
            else:
                explanation_b = json.dumps(explanation_b)

        prompt_ab = create_prompt(name, problem, code, explanation_a, explanation_b, user_inquiry_history)
        prompt_ba = create_prompt(name, problem, code, explanation_a, explanation_b, user_inquiry_history, reverse=True)

        response_ab = text_model([HumanMessage(content=prompt_ab)])
        response_ba = text_model([HumanMessage(content=prompt_ba)])
        
        
        try:
            response_ab_json = json.loads(response_ab.content)
            response_ba_json = json.loads(response_ba.content)
            return response_ab_json, response_ba_json
        except json.JSONDecodeError as e:
            return None

        
    while True:
        result = get_explanations_and_decisions()
        if result is not None:
            return result
        time.sleep(1)  # Wait for 1 second before retrying


with multiprocessing.Pool(processes=10) as pool:
    results = pool.map(get_explanations_and_decisions_with_retry, target_idx_list)

def parse_decision(response_json):
    decision = response_json["leaderboard"]
    return decision

a_wins = 0
b_wins = 0
ties = 0

for response_ab_json, response_ba_json in results:
    decision_ab = parse_decision(response_ab_json)
    decision_ba = parse_decision(response_ba_json)

    if decision_ab[0] == "method_0" and decision_ba[0] == "method_1":
        a_wins += 1
    elif decision_ab[0] == "method_1" and decision_ba[0] == "method_0":
        b_wins += 1
    else:
        ties += 1

total_comparisons = len(results)
a_win_rate = a_wins / total_comparisons
b_win_rate = b_wins / total_comparisons
tie_rate = ties / total_comparisons

print(f"A win rate: {a_win_rate}")
print(f"B win rate: {b_win_rate}")
print(f"Tie rate: {tie_rate}")


