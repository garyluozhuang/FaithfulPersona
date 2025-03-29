import pandas as pd
import random

random.seed(42)

data = pd.read_csv('QueryResults.csv')
print(data.head())

data['CreationDate'] = pd.to_datetime(data['CreationDate'])

user_question_count = data.groupby('OwnerUserId').size()

eligible_users = user_question_count[user_question_count >= 5].index

selected_users = random.sample(list(eligible_users), 10)

for user in selected_users:
    user_data = data[data['OwnerUserId'] == user].sort_values(by='CreationDate', ascending=False).head(5)
    user_questions = user_data[['Title', 'Tags', 'Body', 'Answer']]
    
    file_name = f'user_{user}_ground_truth_tag.txt'
    with open(file_name, 'w', encoding='utf-8') as file:
        for idx, row in user_questions.iterrows():
            file.write(f"Title: {row['Title']}\n")
            file.write(f"Tags: {row['Tags']}\n")
            file.write(f"Body: {row['Body']}\n\n")
    
    file_name = f'user_{user}_questions.txt'
    with open(file_name, 'w', encoding='utf-8') as file:
        for idx, row in user_questions.iterrows():
            file.write(f"Title: {row['Title']}\n")
            file.write(f"Body: {row['Body']}\n\n")

    file_name = f'user_{user}_ground_truth.txt'
    with open(file_name, 'w', encoding='utf-8') as file:
        for idx, row in user_questions.iterrows():
            file.write(f"Title: {row['Title']}\n")
            file.write(f"Tags: {row['Tags']}\n")
            file.write(f"Body: {row['Body']}\n\n")
            file.write(f"Answer: {row['Answer']}\n\n")

    
