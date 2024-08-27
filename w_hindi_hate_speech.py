import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

test_data_path = r'D:\ML-Test\dataset-hindi-hate-speech\hindi_hate_test.csv'
test_df = pd.read_csv(test_data_path)

label_mapping = {'hateful': 1, 'not_hateful': 0}  # Adjust as per your dataset
test_df['label'] = test_df['label_gold'].map(label_mapping)

test_df = test_df.dropna(subset=['label'])

test_df['label'] = test_df['label'].astype(int)

def preprocess_text(text):
    return str(text).strip()

test_df['test_case'] = test_df['test_case'].apply(preprocess_text)

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

model_name = 'google/muril-base-cased'
model = AutoModelForSequenceClassification.from_pretrained(model_name)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def predict_text(model, tokenizer, text, max_len=128, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
        prediction = logits.argmax(dim=1).item()

    label = 'Hate Speech' if prediction == 1 else 'Not Hate Speech'
    confidence = probabilities[prediction] * 100

    return label, confidence

sample_text_non_hate = "हम सबको एक साथ मिलकर काम करना चाहिए।"
label, confidence = predict_text(model, tokenizer, sample_text_non_hate)
print(f'Text: {sample_text_non_hate}')
print(f'Prediction: {label} (Confidence: {confidence:.2f}%)\n')

sample_text_hate = "इन लोगों से नफरत करना सही है।"
label, confidence = predict_text(model, tokenizer, sample_text_hate)
print(f'Text: {sample_text_hate}')
print(f'Prediction: {label} (Confidence: {confidence:.2f}%)\n')

sample_text_non_hate_positive = "मुझे आपसे प्यार है।"
label, confidence = predict_text(model, tokenizer, sample_text_non_hate_positive)
print(f'Text: {sample_text_non_hate_positive}')
print(f'Prediction: {label} (Confidence: {confidence:.2f}%)\n')

sample_text_hate_aggressive = "तुम लोग बहुत गंदे हो।"
label, confidence = predict_text(model, tokenizer, sample_text_hate_aggressive)
print(f'Text: {sample_text_hate_aggressive}')
print(f'Prediction: {label} (Confidence: {confidence:.2f}%)\n')

