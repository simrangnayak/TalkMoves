import pandas as pd
import numpy as np

# see all rows and columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# set seed for reproducibility
np.random.seed(42)

# use 7 context window
context_window_size = 7

# Load the NCTE single utterances CSV
print("Loading ncte_single_utterances.csv...")
df = pd.read_csv('ncte_single_utterances.csv')

# Drop rows where speaker is NA
df = df.dropna(subset=['speaker'])

print(f"Loaded data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Unique OBSIDs: {df['OBSID'].nunique()}")

# Create speaker flag: 'T' for teacher, 'S' for student or multiple students
df['speaker_flag'] = df['speaker'].apply(lambda x: 'T' if x.lower() == 'teacher' else 'S')

# Group by OBSID
utterances_list = []
for obsid, group in df.groupby('OBSID'):
    group = group.reset_index(drop=True)
    
    for i in range(len(group)):
        # Only process student utterances
        speaker_flag = group.loc[i, "speaker_flag"]
        if speaker_flag != 'S':
            continue
        
        utterance = group.loc[i, "text"]
        turn_val = group.loc[i, "turn_idx"]
        
        # Try to convert to int, keep original if it fails
        if pd.isna(turn_val):
            turn_tag = np.nan
        else:
            try:
                turn_tag = int(turn_val)
            except (ValueError, TypeError):
                turn_tag = turn_val
        
        # Build previous context (up to context_window_size utterances)
        previous = []
        previous_counts = 0
        for j in range(i - context_window_size, i):
            if j < 0:
                continue
            prev_utterance = group.loc[j, "text"]
            prev_speaker = group.loc[j, "speaker_flag"]
            prev_val = group.loc[j, "turn_idx"]
            # Try to convert to int, keep original if it fails
            if pd.isna(prev_val):
                prev_turn = np.nan
            else:
                try:
                    prev_turn = int(prev_val)
                except (ValueError, TypeError):
                    prev_turn = prev_val
            previous.append(f"({prev_turn}) [{prev_speaker}] {prev_utterance}")
            previous_counts += 1
        
        # Build subsequent context (up to context_window_size utterances)
        subsequent = []
        subsequent_counts = 0
        for j in range(i + 1, i + context_window_size + 1):
            if j >= len(group):
                continue
            sub_utterance = group.loc[j, "text"]
            sub_speaker = group.loc[j, "speaker_flag"]
            sub_val = group.loc[j, "turn_idx"]
            # Try to convert to int, keep original if it fails
            if pd.isna(sub_val):
                sub_turn = np.nan
            else:
                try:
                    sub_turn = int(sub_val)
                except (ValueError, TypeError):
                    sub_turn = sub_val
            subsequent.append(f"({sub_turn}) [{sub_speaker}] {sub_utterance}")
            subsequent_counts += 1
        
        # Only include if we have full context windows
        if previous_counts < context_window_size or subsequent_counts < context_window_size:
            continue
        
        previous_str = "\n".join(previous)
        subsequent_str = "\n".join(subsequent)
        
        utterances_list.append({
            'previous_context': previous_str,
            'student_utterance': utterance,
            'turn': turn_tag,
            'subsequent_context': subsequent_str,
            'OBSID': obsid,
            'speaker': group.loc[i, "speaker"],
            'speaker_flag': speaker_flag,
            'year': group.loc[i, "year"]
        })

print(f"Created {len(utterances_list)} utterance records with full context windows")

# Create DataFrame
df_utterances = pd.DataFrame(utterances_list)

print(f"Final dataframe shape: {df_utterances.shape}")
print(f"\nSpeaker distribution:")
print(df_utterances['speaker'].value_counts())

# Save to CSV
output_path = 'ncte_utterances_with_context.csv'
df_utterances.to_csv(output_path, index=False)
print(f"\nSaved to {output_path}")

# Also display sample
print("\nSample record:")
if len(df_utterances) > 0:
    print(df_utterances.iloc[0])
