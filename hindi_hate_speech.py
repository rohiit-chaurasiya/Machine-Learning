import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

train_data_path = 'D:\ML-Test\dataset-hindi-hate-speech\hindi_hate_train.csv'
test_data_path = 'D:\ML-Test\dataset-hindi-hate-speech\hindi_hate_test.csv'

train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

label_mapping = {'HOF': 1, 'NOT': 0}
train_df['label'] = train_df['task_1'].map(label_mapping)
test_df['label'] = test_df['task_1'].map(label_mapping)

train_df = train_df[['text', 'label']].dropna()
test_df = test_df[['text', 'label']].dropna()


def preprocess_text(text):
    return str(text).strip()


train_df['text'] = train_df['text'].apply(preprocess_text)
test_df['text'] = test_df['text'].apply(preprocess_text)

from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(train_df['label']),
                                     y=train_df['label'].values)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
print(f"Class Weights: {class_weights}")


class HASOCDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Create data loaders
RANDOM_SEED = 42
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 4
LEARNING_RATE = 2e-5

tokenizer = AutoTokenizer.from_pretrained('google/muril-base-cased')

train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_df['text'],
    train_df['label'],
    test_size=0.1,
    random_state=RANDOM_SEED
)

train_dataset = HASOCDataset(
    texts=train_texts.values,
    labels=train_labels.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

val_dataset = HASOCDataset(
    texts=val_texts.values,
    labels=val_labels.values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = HASOCDataset(
    texts=test_df['text'].values,
    labels=test_df['label'].values,
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

model = AutoModelForSequenceClassification.from_pretrained(
    'google/muril-base-cased',
    num_labels=2
)
model = model.to(device)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights).to(device)


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model.train()

    losses = []
    correct_predictions = 0

    for batch in tqdm(data_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_fn, device, n_examples):
    model.eval()

    losses = []
    correct_predictions = 0
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return correct_predictions.double() / n_examples, np.mean(losses), all_labels, all_preds


history = {
    'train_acc': [],
    'train_loss': [],
    'val_acc': [],
    'val_loss': []
}

best_f1 = 0

for epoch in range(EPOCHS):
    print(f'Epoch {epoch + 1}/{EPOCHS}')
    print('-' * 20)

    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        loss_fn,
        optimizer,
        device,
        scheduler,
        len(train_dataset)
    )

    val_acc, val_loss, val_labels, val_preds = eval_model(
        model,
        val_loader,
        loss_fn,
        device,
        len(val_dataset)
    )

    val_f1 = f1_score(val_labels, val_preds, average='weighted')

    print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
    print(f'Validation loss {val_loss:.4f} accuracy {val_acc:.4f} F1-score {val_f1:.4f}')

    history['train_acc'].append(train_acc)
    history['train_loss'].append(train_loss)
    history['val_acc'].append(val_acc)
    history['val_loss'].append(val_loss)

    if val_f1 > best_f1:
        torch.save(model.state_dict(), 'best_model_state.bin')
        best_f1 = val_f1

model.load_state_dict(torch.load('best_model_state.bin'))

test_acc, test_loss, test_labels, test_preds = eval_model(
    model,
    test_loader,
    loss_fn,
    device,
    len(test_dataset)
)

test_f1 = f1_score(test_labels, test_preds, average='weighted')

print(f'Test loss {test_loss:.4f} accuracy {test_acc:.4f} F1-score {test_f1:.4f}')

report = classification_report(test_labels, test_preds, target_names=['NOT', 'HOF'])
print("Classification Report:")
print(report)

cm = confusion_matrix(test_labels, test_preds)
df_cm = pd.DataFrame(cm, index=['NOT', 'HOF'], columns=['NOT', 'HOF'])

plt.figure(figsize=(6, 4))
sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix on Test Data')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history['train_acc'], label='train_accuracy')
plt.plot(history['val_acc'], label='validation_accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history['train_loss'], label='train_loss')
plt.plot(history['val_loss'], label='validation_loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.show()


def predict_text(model, tokenizer, text, max_len=MAX_LEN):
    model.eval()
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        logits = outputs.logits
        _, prediction = torch.max(logits, dim=1)
        proba = torch.softmax(logits, dim=1).cpu().numpy()[0]

    label = 'Hate Speech' if prediction == 1 else 'Not Hate Speech'
    confidence = proba[prediction] * 100

    return label, confidence


sample_text = "यह एक परीक्षण वाक्य है।"
label, confidence = predict_text(model, tokenizer, sample_text)
print(f'Text: {sample_text}')
print(f'Prediction: {label} (Confidence: {confidence:.2f}%)')
