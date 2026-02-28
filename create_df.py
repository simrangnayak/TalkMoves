import pandas as pd
import numpy as np
import glob

# see all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# set seed for reproducibility
np.random.seed(42)

# use 7 context window
context_window_size = 7

student_tag_dict = {
    '4 - Making a Claim': '4 - Making a Claim',
    '5 - Providing Evidence/Explaining Reasoning': '5 - Providing Evidence/Explaining Reasoning',
    '1 - None': '1 - None',
    '2 - Relating to Another S': '2 - Relating to Another Student',
    '5 - Providing Evidence / Explaining Reasoning': '5 - Providing Evidence/Explaining Reasoning',
    '3 - Asking for More Information': '3 - Asking for More Information',
    '2 - Relating to Another Student': '2 - Relating to Another Student',
    '4 - making a claim': '4 - Making a Claim',
    '5 - providing evidence/providing reasoning': '5 - Providing Evidence/Explaining Reasoning',
    '2 - relating to another student': '2 - Relating to Another Student',
    '1 - none': '1 - None',
    '3 - asking for more information': '3 - Asking for More Information',
    2: '2 - Relating to Another Student',
    '2 - relating to another S': '2 - Relating to Another Student',
    '3 - Asking for Information': '3 - Asking for More Information',
    5: '5 - Providing Evidence/Explaining Reasoning',
    1: '1 - None',
    '\\': '1 - None',
    '4 - making a cDavid': '4 - Making a Claim',
    ' ': '1 - None',
    '5 - providing evidence / reasoning': '5 - Providing Evidence/Explaining Reasoning'
}

# Find all xlsx files in Subset 1 and Subset 2
xlsx_files = []
xlsx_files.extend(glob.glob('data/Subset 1/**/*.xlsx', recursive=True))
xlsx_files.extend(glob.glob('data/Subset 2/**/*.xlsx', recursive=True))

# Filter out temporary files (starting with ~)
xlsx_files = [f for f in xlsx_files if not f.split('/')[-1].startswith('~')]

# Separate into train and test based on teacher names
train_xlsx = []
test_xlsx = []
for item in xlsx_files:
    if item.find("Travers.Spring") != -1 or item.find("Benson") != -1 \
    or item.find("Carroll") != -1 or item.find("Saunders.Spring") != -1 \
    or item.find("Keene") != -1 or item.find("Carter") != -1 or item.find("Basker") != -1 :
        test_xlsx.append(item)
    else:
        train_xlsx.append(item)

print(f"Found {len(train_xlsx)} training files and {len(test_xlsx)} test files")

# Load and concatenate all files
temp = []
for file in train_xlsx:
    try:
        load_data = pd.read_excel(file)
        # Add filename column (only the filename, not the full path)
        load_data['filename'] = file.split('/')[-1]
        
        # create a column for if speaker is student or not - if Student Tag is not null, then it is a student utterance, and convert to "T" else "S"
        load_data['is_student'] = load_data['Student Tag'].notna()
        load_data['student_tag'] = load_data['Student Tag'].map(student_tag_dict)
        load_data['speaker_flag'] = load_data['is_student'].apply(lambda x: 'S' if x else 'T')
        
        utterances_list = []
        for i in range(len(load_data)):
            utterance = load_data.loc[i, "Sentence"]
            student_tag = load_data.loc[i, "student_tag"]
            turn_val = load_data.loc[i, "Turn"]
            
            # Try to convert to int, keep original if it fails
            if pd.isna(turn_val):
                turn_tag = np.nan
            else:
                try:
                    turn_tag = int(turn_val)
                except (ValueError, TypeError):
                    turn_tag = turn_val

            if pd.isna(student_tag):
                continue  # Skip rows without a student tag
                
            previous = []
            previous_counts = 0
            subsequent = []
            subsequent_counts = 0
            for j in range(i-context_window_size, i):
                if j < 0:
                    continue
                prev_utterance = load_data.loc[j, "Sentence"]
                prev_speaker = load_data.loc[j, "speaker_flag"]
                prev_val = load_data.loc[j, "Turn"]
                try:
                    prev_tag = int(prev_val) if pd.notna(prev_val) else prev_val
                except (ValueError, TypeError):
                    prev_tag = prev_val
                previous.append(f"({prev_tag}) [{prev_speaker}] {prev_utterance}")
                previous_counts += 1
            for j in range(i+1, i+context_window_size+1):
                if j >= len(load_data):
                    continue
                sub_utterance = load_data.loc[j, "Sentence"]
                sub_speaker = load_data.loc[j, "speaker_flag"]
                sub_val = load_data.loc[j, "Turn"]
                try:
                    sub_tag = int(sub_val) if pd.notna(sub_val) else sub_val
                except (ValueError, TypeError):
                    sub_tag = sub_val
                subsequent.append(f"({sub_tag}) [{sub_speaker}] {sub_utterance}")
                subsequent_counts += 1
            if previous_counts < context_window_size:
                continue  # Skip if previous context_window_size utterances are not complete
            if subsequent_counts < context_window_size:
                continue  # Skip if subsequent context_window_size utterances are not complete
            previous_str = "\n".join(previous)
            subsequent_str = "\n".join(subsequent)
            utterances_list.append({
                'previous_context': previous_str,
                'student_utterance': utterance,
                'subsequent_context': subsequent_str,
                'filename': file.split('/')[-1],
                'student_tag': student_tag,
                'turn': turn_tag
            })

        df_utterances = pd.DataFrame(utterances_list)
        temp.append(df_utterances)
    except Exception as e:
        print(f"Error loading {file}: {e}")

if temp:
    train_df = pd.concat(temp, ignore_index=True)
    print(f"\nCombined dataframe shape: {train_df.shape}")
else:
    print("No files were loaded!")


print(train_df.head())

train_df = train_df[['previous_context', 'student_utterance', 'turn', 'subsequent_context', 'filename', 'student_tag']]

sampled_dfs = []
for tag, count in [('2 - Relating to Another Student', 60), ('3 - Asking for More Information', 50), ('4 - Making a Claim', 25), ('5 - Providing Evidence/Explaining Reasoning', 25), ('1 - None', 20)]:
    sampled_df = train_df[train_df['student_tag'] == tag].sample(n=count, random_state=42)
    sampled_dfs.append(sampled_df)

full_train_df_sampled = pd.concat(sampled_dfs, ignore_index=True)
full_train_df_unannotated = train_df[~train_df.index.isin(full_train_df_sampled.index)]
full_train_df_sampled = full_train_df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(full_train_df_sampled['student_tag'].value_counts())
full_train_df_sampled.to_csv('talk_moves_annotate_new.csv', index=False)
full_train_df_unannotated.to_csv('talk_moves_unannotated.csv', index=False)
