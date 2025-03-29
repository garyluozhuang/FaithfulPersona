from datasets import load_dataset
import pandas as pd

group = "valid"
data = load_dataset("deepmind/code_contests", split=group)
df = pd.DataFrame(data)

results = []

target_idx_list = [
    idx for idx, value in df.iterrows() if "<image>" not in value["description"] and
    (3 in value["solutions"]["language"] or 1 in value["solutions"]["language"])
]

for idx in target_idx_list:
    df_sample = df.loc[idx, :]
    problem = df_sample["description"]
    name = df_sample["name"]

    solutions = [soln for lang, soln in zip(df_sample["solutions"]["language"], df_sample["solutions"]["solution"]) if lang in (1, 3)]
    code = min(solutions, key=len)

    results.append({"problem": problem, "name": name, "code": code})

result_df = pd.DataFrame(results)

result_df.to_csv("problems_solutions_valid.csv", index=False)
