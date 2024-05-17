import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaForSequenceClassification, AdamW
from tqdm import tqdm

# Load the CSV file
df = pd.read_csv('Roberta_model1.csv')

# Split into train, validation, and test sets
train_df, test_val_df = train_test_split(df, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(test_val_df, test_size=0.5, random_state=42)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)
# Define a custom dataset class
class CustomDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# Define hyperparameters
MAX_LENGTH = 128
BATCH_SIZE = 32
LEARNING_RATE = 2e-5
EPOCHS = 5

# Initialize RoBERTa tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)

# Create DataLoader for train, validation, and test sets
train_dataset = CustomDataset(train_df['Text'], train_df['Label'], tokenizer, MAX_LENGTH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CustomDataset(val_df['Text'], val_df['Label'], tokenizer, MAX_LENGTH)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

test_dataset = CustomDataset(test_df['Text'], test_df['Label'], tokenizer, MAX_LENGTH)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define optimizer and loss function
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
criterion = torch.nn.CrossEntropyLoss()

# Fine-tune the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
validation_accuracies = []
for epoch in range(EPOCHS):
    # Training loop
    model.train()
    train_loss = 0.0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='Validation:'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            val_loss += loss.item()

            _, predicted = torch.max(outputs.logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = correct / total
    validation_accuracies = []

    print(f'Epoch {epoch + 1}/{EPOCHS}, Val Accuracy: {val_accuracy:.2%}')
    with open('validation_accuracies_gpt.txt', 'a') as f:
        f.write(f'Epoch {epoch + 1}: {val_accuracy:.2%}\n')

# Save the list of validation accuracies to a file
with open('validation_accuracies_gpt_list.txt', 'w') as f:
    for epoch, accuracy in enumerate(validation_accuracies, start=1):
        f.write(f'Epoch {epoch}: {accuracy:.2%}\n')


# Testing loop
model.eval()
test_loss = 0.0
correct = 0
total = 0
predicted_labels = []
true_labels = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing:'):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        test_loss += loss.item()

        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Collect predicted and true labels
        predicted_labels.extend(predicted.cpu().numpy())
        true_labels.extend(labels.cpu().numpy())

avg_test_loss = test_loss / len(test_loader)
test_accuracy = correct / total

precision = precision_score(true_labels, predicted_labels)

print(f'Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2%}, Precision: {precision:.2%}')

# Save results
results = {
    'Test Loss': avg_test_loss,
    'Test Accuracy': test_accuracy,
    'Precision': precision
}

# Save results to a file
with open('results_gpt.txt', 'w') as f:
    for key, value in results.items():
        f.write(f'{key}: {value}\n')
