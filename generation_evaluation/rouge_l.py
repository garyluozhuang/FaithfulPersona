
import argparse
import json
from datasets import load_dataset
import pandas as pd
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import spacy
from rouge_score import rouge_scorer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Ensure the spaCy model is downloaded
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Initialize NLP tools
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--group', type=str, default='valid', help='Group name (default: valid)')
    parser.add_argument('--sample_num', type=int, default=1, help='Sample number (default: 1)')
    parser.add_argument('--user_id', type=int, default=179736, help='User ID (default: 179736)')
    parser.add_argument('--method', type=str, default='base_personalization', help='Method (default: base_personalization)')
    parser.add_argument('--iter_count', type=int, default=1, help='Iteration count (default: 1)')
    return parser.parse_args()

def load_user_inquiry_history(user_id):
    with open(f'user_{user_id}_ground_truth.txt', 'r') as file:
        return file.read()

def get_folder_path(args):
    if args.method == "base_personalization":
        return f"./result/explanation/{args.group}/base_personalization/{args.user_id}/sample={args.sample_num}/"
    elif args.method == "base_personalization_consistency":
        return f"./result/explanation/{args.group}/base_personalization_consistency/{args.user_id}/sample={args.sample_num}/"
    elif args.method == "disco":
        return f"./result/explanation/{args.group}/exp_disco/iter={args.iter_count}/sample={args.sample_num}/"
    elif args.method == "disco_personalization":
        return f"./result/explanation/{args.group}/exp_disco_personalization/{args.user_id}/iter={args.iter_count}/sample={args.sample_num}/"

def calculate_rouge_l(explanation, user_inquiry_history):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(user_inquiry_history, explanation)
    return scores['rougeL'].fmeasure

def main():
    args = parse_arguments()
    folder_path = get_folder_path(args)
    user_inquiry_history = load_user_inquiry_history(args.user_id)

    data = load_dataset("deepmind/code_contests", split=args.group)
    df = pd.DataFrame(data)

    target_idx_list = [
        idx for idx, value in df.iterrows() if "<image>" not in value["description"] and
        (3 in value["solutions"]["language"] or 1 in value["solutions"]["language"])
    ]

    total_rouge_l = 0
    count = 0

    for idx in target_idx_list:
        df_sample = df.loc[idx, :]
        name = df_sample["name"]
        
        explanation_path = os.path.join(folder_path, name, "best_explanation.txt")
        if os.path.exists(explanation_path):
            with open(explanation_path, "r") as file:
                explanation = file.read()
                explanation = json.loads(explanation)
                if "personalized_explanation" in explanation:
                    explanation = {"personalized_explanation": explanation["personalized_explanation"]}
                    explanation = json.dumps(explanation)
                else:
                    explanation = json.dumps(explanation)
            rouge_l_score = calculate_rouge_l(explanation, user_inquiry_history)
            total_rouge_l += rouge_l_score
            count += 1

    if count > 0:
        average_rouge_l = total_rouge_l / count
        print(f'Average ROUGE-L score: {average_rouge_l:.4f}')
    else:
        print('No valid explanations found.')

if __name__ == "__main__":
    main()

