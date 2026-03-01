import pandas as pd
import numpy as np

# annotated_df = pd.read_csv("talk_moves_annotate - Sheet2.csv")
# df = pd.read_csv("talk_moves_annotate_new.csv")
# unannotated_df = pd.read_csv("talk_moves_unannotated.csv")
# unannotated_df = unannotated_df.drop(columns=['filename', 'student_tag'])

# df2 = pd.merge(df, annotated_df[['student_utterance', 'turn', 'Offering Math Help','Successful Uptake','filename']], on=['student_utterance', 'turn', 'filename'], how='left')
# df2 = df2.drop(columns=['filename', 'student_tag'])
# print(df2.columns)
# print(unannotated_df.columns)
# df2.to_csv("talk_moves_validation_set.csv", index=False)
# unannotated_df.to_csv("talk_moves_train_set.csv", index=False)

train = pd.read_csv("talk_moves_train_set.csv")
validation = pd.read_csv("talk_moves_validation_set.csv")
print(train.shape)
print(validation.shape)