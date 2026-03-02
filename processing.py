import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_csv('ncte_utterances_with_context.csv')
df = df[['previous_context','student_utterance','turn','subsequent_context']]

np.random.seed(42)
# sample 200 rows for annotation
sampled_df = df.sample(n=200, random_state=42).reset_index(drop=True)
sampled_df.to_csv('ncte_utterances_sampled_for_annotation.csv', index=False)
print(sampled_df.head())


#previous_context,student_utterance,turn,subsequent_context