import pandas as pd
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch import nn
import warnings
from transformers import logging as hf_logging

hf_logging.set_verbosity_error()

np.random.seed(42)
torch.manual_seed(42)

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
train_df = pd.read_csv('LLM_annotations/llm_annotated_train_set.csv')
test_df = pd.read_csv('talk_moves_validation_set.csv')

# Helper functions for data processing
def get_turn_num(row):
    if pd.notna(row['turn']):
        try:
            return int(float(row['turn']))
        except ValueError:
            return 'nan'
    return 'nan'

def create_input_text(row):
    turn_num = get_turn_num(row)
    
    prev   = row['previous_context']   if pd.notna(row['previous_context'])   else '(none)'
    subseq = row['subsequent_context'] if pd.notna(row['subsequent_context']) else '(none)'
    
    seq_a = prev
    seq_b = f"[TARGET_START] ({turn_num}) [S] {row['student_utterance']} [TARGET_END] {subseq}"
    
    return seq_a, seq_b


def make_dataset(df):
    seq_a, seq_b = zip(*df.apply(create_input_text, axis=1))
    return Dataset.from_dict({
        'seq_a': list(seq_a),
        'seq_b': list(seq_b),
        'label': df['label'].tolist()
    })


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


def run_model(label):
    print(f"\n{'='*60}")
    print(f"Training model for: {label}")
    print(f"{'='*60}")
    
    # Calculate class weights
    neg_count = (train_df[label] == 0).sum()
    pos_count = (train_df[label] == 1).sum()
    pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
    print(f"Class weight for positive class: {pos_weight:.2f}")
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)

    print(f"Test set size: {len(test_df)}")
    print(f"Positive examples in test: {test_df[label].sum()}")
    print(f"Negative examples in test: {(test_df[label]==0).sum()}")
    print(f"Positive ratio in test: {test_df[label].mean():.3f}")

    print(f"\nTrain set size: {len(train_df)}")
    print(f"Positive examples in train: {train_df[label].sum()}")
    print(f"Positive ratio in train: {train_df[label].mean():.3f}")

    # Load model and tokenizer
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)

    # Add special tokens for target utterance boundary
    special_tokens = {'additional_special_tokens': ['[TARGET_START]', '[TARGET_END]']}
    tokenizer.add_special_tokens(special_tokens)

    model = RobertaForSequenceClassification.from_pretrained(model_name, 
                                                             num_labels=2,
                                                             classifier_dropout=0.2,
                                                             hidden_dropout_prob=0.1)
    model.resize_token_embeddings(len(tokenizer))

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Prepare datasets
    train_data_label = train_df[['previous_context', 'student_utterance', 'turn', 'subsequent_context', label]].copy()
    train_data_label.columns = ['previous_context', 'student_utterance', 'turn', 'subsequent_context', 'label']
    
    test_data_label = test_df[['previous_context', 'student_utterance', 'turn', 'subsequent_context', label]].copy()
    test_data_label.columns = ['previous_context', 'student_utterance', 'turn', 'subsequent_context', 'label']
    
    train_dataset = make_dataset(train_data_label)
    test_dataset = make_dataset(test_data_label)

    # Check truncation
    lengths = pd.Series([
        len(tokenizer(a, b)['input_ids'])
        for a, b in zip(train_dataset['seq_a'], train_dataset['seq_b'])
    ])
    print(f"\nSequence length statistics:")
    print(lengths.describe())
    print(f"Truncated (>512): {(lengths > 512).sum()} / {len(lengths)} ({100*(lengths > 512).mean():.1f}%)")

    def tokenize_function(examples):
        return tokenizer(
            examples['seq_a'],
            examples['seq_b'],
            truncation=True,
            max_length=512
        )

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

    # Set up training arguments
    label_safe = label.replace(' ', '_')
    training_args = TrainingArguments(
        output_dir=f'./roberta_model_{label_safe}',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.06,
        weight_decay=0.05,
        learning_rate=2e-5,
        logging_dir=f'./logs_{label_safe}',
        logging_steps=50,
        eval_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        metric_for_best_model='f1',
        greater_is_better=True,
        use_mps_device=True
    )

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator
    )

    print(f"\nStarting training for {label}...")
    trainer.train()

    print(f"\nEvaluating on test set...")
    results = trainer.evaluate(test_dataset)
    print(f"Test Results for {label}:")
    for key, value in results.items():
        print(f"  {key}: {value:.4f}")

    # Get and save predictions
    predictions = trainer.predict(test_dataset)
    pred_labels = np.argmax(predictions.predictions, axis=1)
    test_data_label['predicted_label'] = pred_labels
    test_data_label.to_csv(f'test_predictions_{label_safe}.csv', index=False)

    model.save_pretrained(f'./roberta_model_{label_safe}')
    tokenizer.save_pretrained(f'./roberta_model_{label_safe}')
    print(f"\nCompleted training for {label}.\n")


if __name__ == "__main__":
    labels = ['Offering Math Help', 'Successful Uptake']

    for label in labels:
        run_model(label)
