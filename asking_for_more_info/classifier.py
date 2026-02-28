import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

train_df = pd.read_csv('full_train_df.csv')
test_df = pd.read_csv('full_test_df.csv')
train_df['label'] = train_df['student_tag'].apply(lambda x: 1 if x == '3 - Asking for More Information' else 0)
test_df['label'] = test_df['student_tag'].apply(lambda x: 1 if x == '3 - Asking for More Information' else 0)
train_df = train_df[['previous_context', 'student_utterance', 'turn','subsequent_context', 'label']]
test_df = test_df[['previous_context', 'student_utterance', 'turn','subsequent_context', 'label']]
print(train_df.head())
print(test_df.head())