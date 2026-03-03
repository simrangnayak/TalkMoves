import pandas as pd
import numpy as np

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.read_csv('fully_labeled_ncte_dataset.csv')
print(df['Predicted_Math_Help'].value_counts())
print(df['Predicted_Successful_Uptake'].value_counts())

#previous_context,student_utterance,turn,subsequent_context