import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import csv

# Your Hindi data
new_data = [
    "आपका दिन अच्छा हो!",
    "आप बहुत अच्छे इंसान हैं।",
    "आज मौसम बहुत अच्छा है।",
    "मैं बाजार जा रहा हूँ।",
    "मुझे यह फिल्म बिल्कुल पसंद नहीं आई।",
    "यह देरी बहुत परेशान करने वाली है।",
    "मैं तुमसे नफरत करता हूँ, तुम गंदे लोग हो!",
    "यहाँ से निकल जाओ, तुम सब कचरा हो!",
    "तुम बेवकूफ हो, कुछ समझ नहीं आता क्या?",
    "ये लोग बेकार हैं, शर्मनाक!"
]

# Simulated labeled dataset (for demonstration; replace with real data)
# Labels: 0 = No Hate, 1 = Weakly Hateful, 2 = Strongly Hateful
train_data = [
    ("आप बहुत अच्छे हैं।", 0),
    ("मौसम अच्छा है।", 0),
    ("मुझे यह पसंद नहीं।", 0),
    ("तुम गंदे हो।", 1),
    ("तुम बेवकूफ हो।", 1),
    ("मैं तुमसे नफरत करता हूँ।", 2),
    ("यहाँ से निकल जाओ, कचरा!", 2),
    ("ये लोग शर्मनाक हैं।", 1),
    ("तुम सब कुत्ते हो!", 2),
    ("सब बेकार हैं।", 1),
    
    # Additional samples
    ("आज का दिन बहुत सुंदर है।", 0),
    ("मुझे खुशी हो रही है।", 0),
    ("तुम एक शानदार इंसान हो।", 0),
    ("इस जगह का नज़ारा बहुत अच्छा है।", 0),
    ("तुम्हारी मदद के लिए धन्यवाद।", 0),
    ("मुझे नहीं लगता कि यह सही है।", 0),
    ("हम यहाँ अच्छा महसूस कर रहे हैं।", 0),
    ("आज बहुत अच्छा दिन है।", 0),
    ("आपकी सलाह उपयोगी है।", 0),
    ("मुझे यह किताब पढ़ना पसंद है।", 0),
    ("तुम बहुत मूर्ख हो।", 1),
    ("ये बहुत बेवकूफी भरी बात है।", 1),
    ("तुमसे बात करना बेकार है।", 1),
    ("तुम कुछ भी ठीक से नहीं कर सकते।", 1),
    ("मुझे तुम पर भरोसा नहीं है।", 1),
    ("तुम हर बार गलत साबित होते हो।", 1),
    ("तुम बिल्कुल भी समझदार नहीं हो।", 1),
    ("तुम्हारी बातों में कोई तर्क नहीं होता।", 1),
    ("तुमसे तो कोई उम्मीद ही नहीं है।", 1),
    ("तुमने फिर से गड़बड़ कर दी।", 1),
    ("भागो यहाँ से, कमीने!", 2),
    ("तुम बहुत घटिया आदमी हो।", 2),
    ("तुम से गिरी हुई इंसानियत नहीं देखी मैंने।", 2),
    ("किसी काम के नहीं हो तुम।", 2),
    ("सबसे बेकार आदमी हो तुम।", 2),
    ("तुमसे ज्यादा निकम्मा कोई नहीं।", 2),
    ("तुम बहुत गिरे हुए इंसान हो।", 2),
    ("किसी काम के लायक नहीं हो।", 2),
    ("निकाल दो इसको यहाँ से!", 2),
    ("तुमसे बात करना भी गवारा नहीं।", 2),
    
    # More neutral phrases
    ("यह खाना बहुत स्वादिष्ट है।", 0),
    ("मुझे संगीत सुनना पसंद है।", 0),
    ("मैं हर दिन योग करता हूँ।", 0),
    ("पढ़ाई करना अच्छा लगता है।", 0),
    ("नया अनुभव हमेशा रोमांचक होता है।", 0),
    
    # More mildly offensive phrases
    ("तुम कभी कुछ नया नहीं सीखते।", 1),
    ("तुम बेकार सुझाव देते हो।", 1),
    ("तुम हमेशा देरी से आते हो।", 1),
    ("तुम्हारी सोच बहुत सीमित है।", 1),
    ("तुमसे कोई भी समझदारी की उम्मीद नहीं कर सकता।", 1),
    
    # More highly offensive phrases
    ("तुम्हारी शक्ल ही गिरी हुई लगती है।", 2),
    ("जितना गंदा इंसान मैंने नहीं देखा।", 2),
    ("तुम्हें यहाँ से भगा देना चाहिए।", 2),
    ("तुम बिल्कुल ही बेकार हो।", 2),
    ("तुमसे बड़ा मूर्ख कोई नहीं।", 2),
    
    # Repeat and shuffle similar patterns to reach 2000+ samples
]

train_df = pd.DataFrame(train_data, columns=["text", "label"])

# Load IndicBERT tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-bert")
model = AutoModelForSequenceClassification.from_pretrained("ai4bharat/indic-bert", num_labels=3)

# Tokenize the training data
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

train_df_tokenized = train_df.copy()
train_encodings = tokenizer(train_df["text"].tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt")
train_labels = torch.tensor(train_df["label"].tolist())

# Create a PyTorch Dataset
class HateSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = HateSpeechDataset(train_encodings, train_labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="no",  # No evaluation set for this example
    save_strategy="epoch",
    load_best_model_at_end=False,
)

# Define a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Fine-tune the model
trainer.train()

# Tokenize and predict on new data
def predict_hate_speech(texts):
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
    return predictions

# Run predictions
predictions = predict_hate_speech(new_data)
label_map = {0: "No Hate", 1: "Weakly Hateful", 2: "Strongly Hateful"}
predicted_labels = [label_map[p.item()] for p in predictions]

# Prepare results
results = []
for i, (text, pred_label) in enumerate(zip(new_data, predicted_labels)):
    # Simulate a score based on softmax probabilities (optional)
    encodings = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.softmax(outputs.logits, dim=-1).max().item()  # Confidence score
    results.append([str(i), text, "unknown", round(probs, 2), pred_label, round(probs, 2)])

# Export to CSV
fields = ['Unique ID', 'Post', 'Labels Set', 'Total Score', 'Hate Label', 'Subjective Hate Label']
with open('output_transformer.csv', 'w', newline='', encoding='utf-8') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(fields)
    csvwriter.writerows(results)

print("Results saved to 'output_transformer.csv'")