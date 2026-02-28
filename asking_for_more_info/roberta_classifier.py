import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
import warnings
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

np.random.seed(42)
torch.manual_seed(42)

# Load data
train_df = pd.read_csv('full_train_df.csv')
test_df = pd.read_csv('full_test_df.csv')

# Create labels
train_df['label'] = (train_df['student_tag'] == '3 - Asking for More Information').astype(int)
test_df['label'] = (test_df['student_tag'] == '3 - Asking for More Information').astype(int)

# Calculate class weights
neg_count = (train_df['label'] == 0).sum()
pos_count = (train_df['label'] == 1).sum()
pos_weight = neg_count / pos_count
print(f"\nClass weight for positive class: {pos_weight:.2f}")
class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)


# Load model and tokenizer
model_name = "roberta-base"
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = RobertaForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Processs data to create input sequences for RoBERTa
def get_turn_num(row):
    if pd.notna(row['turn']):
        try:
            return int(float(row['turn']))
        except ValueError:
            return 'nan'
    return 'nan'


def create_input_text(row):
    turn_num = get_turn_num(row)
    
    prev = row['previous_context'] if pd.notna(row['previous_context']) else '(none)'
    subseq = row['subsequent_context'] if pd.notna(row['subsequent_context']) else '(none)'
    target = f"({turn_num}) [S] {row['student_utterance']}"
    
    # RoBERTa's tokenizer supports passing two sequences, which it separates with </s></s>
    # We use this to create a hard boundary between context and target+subsequent
    sequence_a = f"{prev}"
    sequence_b = f"[TARGET] {target} [AFTER] {subseq}"
    
    return sequence_a, sequence_b



# Prepare datasets
def make_dataset(df):
    seq_a, seq_b = zip(*df.apply(create_input_text, axis=1))
    return Dataset.from_dict({
        'seq_a': list(seq_a),
        'seq_b': list(seq_b),
        'label': df['label'].tolist()
    })

train_dataset = make_dataset(train_df)
test_dataset  = make_dataset(test_df)

lengths = pd.Series([
    len(tokenizer(a, b)['input_ids'])
    for a, b in zip(train_dataset['seq_a'], train_dataset['seq_b'])
])
print(lengths.describe())
print(f"Truncated (>512): {(lengths > 512).sum()} / {len(lengths)} ({100*(lengths > 512).mean():.1f}%)")


def tokenize_function(examples):
    return tokenizer(
        examples['seq_a'],
        examples['seq_b'],
        padding='max_length',
        truncation=True,          
        max_length=512
    )


train_dataset = train_dataset.map(tokenize_function, batched=True)
test_dataset  = test_dataset.map(tokenize_function, batched=True)


train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])



# Custom Trainer with class-weighted loss
class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weights = self.class_weights.to(logits.device)
        loss_fn = nn.CrossEntropyLoss(weight=weights)
        loss = loss_fn(logits, labels)
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, zero_division=0),
        'recall': recall_score(labels, predictions, zero_division=0),
        'f1': f1_score(labels, predictions, zero_division=0)
    }

training_args = TrainingArguments(
    output_dir='./roberta_model',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_ratio=0.1,
    weight_decay=0.01,
    learning_rate=2e-5,
    logging_dir='./logs',
    logging_steps=50,
    eval_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='f1',       # optimize for F1, not loss
    greater_is_better=True,
)

trainer = WeightedTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

print("\nStarting training...")
trainer.train()

print("\nEvaluating on test set...")
results = trainer.evaluate(test_dataset)
print("Test Results:")
for key, value in results.items():
    print(f"  {key}: {value:.4f}")

# Get and save predictions
predictions = trainer.predict(test_dataset)
pred_labels = np.argmax(predictions.predictions, axis=1)
test_df['predicted_label'] = pred_labels
test_df.to_csv('test_predictions.csv', index=False)

model.save_pretrained('./roberta_model_final')
tokenizer.save_pretrained('./roberta_model_final')
print("\nDone.")