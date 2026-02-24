import pandas as pd
import numpy as np
import glob

# see all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# use 7 context window

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

hierarchy_dict = {
    '1 - None': 0,
    '4 - Making a Claim': 1,
    '5 - Providing Evidence/Explaining Reasoning': 2,
    '3 - Asking for More Information': 3,
    '2 - Relating to Another Student': 4
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

        # I want to concatenate "Sentence" by "Turn" and keep the "Student Tag" that is "highest" in the turn. For example, if a turn has 3 sentences and the student tags are "1 - None", "2 - Relating to Another Student", "3 - Asking for More Information", then the concatenated sentence should be "Sentence1 Sentence2 Sentence3" and the student tag should be "3 - Asking for More Information". I will use the "Turn" column to group by and concatenate the sentences, and then take the max student tag for each turn.
        def get_max_tag(tags):
            non_null_tags = [tag for tag in tags if pd.notna(tag)]
            if not non_null_tags:
                return None
            return max(non_null_tags, key=lambda tag: hierarchy_dict[tag])
        
        # fill NaN in Turn with previous turn value (forward fill)
        load_data['Turn'] = load_data['Turn'].ffill()
        grouped_data = load_data.groupby('Turn').agg({
            'Sentence': lambda x: ' '.join(str(s) for s in x),
            'student_tag': get_max_tag
        }).reset_index()

        filename = file.split('/')[-1]
        grouped_data['filename'] = filename
        grouped_data['is_student'] = grouped_data['student_tag'].notna()
        grouped_data['speaker_flag'] = grouped_data['is_student'].apply(lambda x: 'S' if x else 'T')
        
        if filename == "7th grade math.xlsx":
            print(grouped_data.head(20))

        # for each student utterance, i want to find the previous 5 utterances and the subsequent 5 utterances. I then want to create a new df "df_utterances" with the following columns: "student_utterance", "previous_5_utterances", "subsequent_5_utterances", "filename", studnet_tag (which corresponds to the student_utterance column).
        utterances_list = []
        for i in range(len(grouped_data)):
            utterance = grouped_data.loc[i, "Sentence"]
            student_tag = grouped_data.loc[i, "student_tag"]

            if pd.isna(student_tag):
                continue  # Skip rows without a student tag

            # now i want to find the previous 5 utterances and the subsequent 5 utterances. I will use the "speaker_flag" column to attach to the utterances. i.e. it should look like "[S] utterance1 \newline [T] utterance2 \newline [S] utterance3"
            previous_5 = []
            previous_5_counts = 0
            subsequent_5 = []
            subsequent_5_counts = 0
            for j in range(i-5, i):
                if j < 0:
                    continue
                prev_utterance = grouped_data.loc[j, "Sentence"]
                prev_speaker = grouped_data.loc[j, "speaker_flag"]
                previous_5.append(f"[{prev_speaker}] {prev_utterance}")
                previous_5_counts += 1
            for j in range(i+1, i+6):
                if j >= len(grouped_data):
                    continue
                sub_utterance = grouped_data.loc[j, "Sentence"]
                sub_speaker = grouped_data.loc[j, "speaker_flag"]
                subsequent_5.append(f"[{sub_speaker}] {sub_utterance}")
                subsequent_5_counts += 1
            if previous_5_counts < 5:
                continue  # Skip if previous 5 utterances are not complete
            if subsequent_5_counts < 5:
                continue  # Skip if subsequent 5 utterances are not complete
            previous_5_str = "\n".join(previous_5)
            subsequent_5_str = "\n".join(subsequent_5)

            utterances_list.append({
                'student_utterance': utterance,
                'previous_5_utterances': previous_5_str,
                'subsequent_5_utterances': subsequent_5_str,
                'filename': file.split('/')[-1],
                'student_tag': student_tag
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



print(train_df['student_tag'].unique())

print(train_df['student_tag'].value_counts())

train_df.to_csv('full_train_df.csv', index=False)

# select 60 random rows that are '2 - Relating to Another Student', 50 random rows that are '3 - Asking for More Information', 25 random rows that are '4 - Making a Claim', 25 random rows that are '5 - Providing Evidence/Explaining Reasoning', and 20 random rows that are '1 - None' and save them to a new csv file "full_train_df_sampled.csv"

sampled_dfs = []
for tag, count in [('2 - Relating to Another Student', 60), ('3 - Asking for More Information', 50), ('4 - Making a Claim', 25), ('5 - Providing Evidence/Explaining Reasoning', 25), ('1 - None', 20)]:
    sampled_df = train_df[train_df['student_tag'] == tag].sample(n=count, random_state=42)
    sampled_dfs.append(sampled_df)

full_train_df_sampled = pd.concat(sampled_dfs, ignore_index=True)

full_train_df_sampled = full_train_df_sampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(full_train_df_sampled['student_tag'].value_counts())
full_train_df_sampled.to_csv('full_train_df_sampled.csv', index=False)

