import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import DistilBertModel, BertModel, BertTokenizerFast
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import os

class SketchyReader(nn.Module):
    def __init__(self):
        super(SketchyReader, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.classifier(cls_output)
        return logits

class IntensiveReader(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(IntensiveReader, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.start_classifier = nn.Linear(self.bert.config.hidden_size, 1)
        self.end_classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state
        start_logits = self.start_classifier(hidden_states).squeeze(-1)
        end_logits = self.end_classifier(hidden_states).squeeze(-1)
        return start_logits, end_logits

class AnswerVerifier(nn.Module):
    def __init__(self):
        super(AnswerVerifier, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.verifier = nn.Linear(self.bert.config.hidden_size, 2)  # Binary classifier

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # CLS token
        logits = self.verifier(cls_output)
        return logits

class RetroReaderModel(nn.Module):
    def __init__(self):
        super(RetroReaderModel, self).__init__()
        self.sketchy_reader = SketchyReader()
        self.intensive_reader = IntensiveReader()
        self.answer_verifier = AnswerVerifier()

    def forward(self, input_ids, attention_mask, relevance_threshold=0.5):
        # Step 1: Sketchy Reader - Identify relevance
        relevance_logits = self.sketchy_reader(input_ids=input_ids, attention_mask=attention_mask)
        relevance_scores = torch.sigmoid(relevance_logits).squeeze(-1)

        # Mask out examples with low relevance
        relevant_mask = relevance_scores >= relevance_threshold

        # Step 2: Intensive Reader - Generate start and end logits for relevant contexts only
        if relevant_mask.any():
            relevant_input_ids = input_ids[relevant_mask]
            relevant_attention_mask = attention_mask[relevant_mask]
            start_logits, end_logits = self.intensive_reader(
                input_ids=relevant_input_ids,
                attention_mask=relevant_attention_mask
            )
        else:
            start_logits, end_logits = None, None  # No relevant context

        # Step 3: Answer Verifier - Validate the generated answer span
        verifier_logits = self.answer_verifier(input_ids=input_ids, attention_mask=attention_mask)

        return {
            "relevance_logits": relevance_logits,
            "start_logits": start_logits,
            "end_logits": end_logits,
            "verifier_logits": verifier_logits,
            "relevant_mask": relevant_mask
        }



dataset = load_dataset("squad")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def preprocess_function(examples):
    tokenized_examples = tokenizer(
        examples["question"],
        examples["context"],
        truncation="only_second",
        padding="max_length",
        max_length=384,
        stride=128,
        return_offsets_mapping=True
    )
    
    start_positions = []
    end_positions = []
    
    for i, offsets in enumerate(tokenized_examples["offset_mapping"]):
        answer = examples["answers"][i]
        start_char = answer["answer_start"][0]
        end_char = start_char + len(answer["text"][0])

        start_position = 0
        end_position = 0
        
        for idx, (start, end) in enumerate(offsets):
            if start <= start_char < end:
                start_position = idx
            if start < end_char <= end:
                end_position = idx
                break

        start_positions.append(start_position)
        end_positions.append(end_position)

    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    tokenized_examples.pop("offset_mapping")
    
    return tokenized_examples

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "start_positions", "end_positions", "id"])

train_loader = DataLoader(tokenized_datasets["train"].select(range(5000)), batch_size=8, shuffle=True)
val_loader = DataLoader(tokenized_datasets["validation"], batch_size=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RetroReaderModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=3e-5)

def train_epoch(model, dataloader, optimizer, relevance_threshold=0.5):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        start_positions = batch['start_positions'].to(device)
        end_positions = batch['end_positions'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, relevance_threshold=relevance_threshold)

        relevant_mask = outputs["relevant_mask"]
        
        if relevant_mask.any():
            relevant_start_logits = outputs["start_logits"]
            relevant_end_logits = outputs["end_logits"]
            relevant_start_positions = start_positions[relevant_mask]
            relevant_end_positions = end_positions[relevant_mask]
            start_loss = F.cross_entropy(relevant_start_logits, relevant_start_positions)
            end_loss = F.cross_entropy(relevant_end_logits, relevant_end_positions)
            loss = start_loss + end_loss

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        else:
            continue

    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, epochs=10):
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")

def save_model(model, tokenizer, model_path="retro_reader_model.pth", tokenizer_path="tokenizer/"):
    torch.save(model.state_dict(), model_path)
    tokenizer.save_pretrained(tokenizer_path)
    print(f"Model saved to {model_path}")
    print(f"Tokenizer saved to {tokenizer_path}")

def load_model(model_path="retro_reader_model.pth", tokenizer_path="tokenizer/"):
    model = RetroReaderModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path)
    print("Model and tokenizer loaded successfully.")
    return model, tokenizer

def generate_predictions(model, dataloader, tokenizer, output_file="predictions.json"):
    model.eval()
    predictions = {}
    model.to(device)
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating Predictions"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            example_ids = batch['id']
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs["start_logits"]
            end_logits = outputs["end_logits"]

            for i, example_id in enumerate(example_ids):
                start_index = torch.argmax(start_logits[i]).item()
                end_index = torch.argmax(end_logits[i]).item()
                answer_tokens = input_ids[i][start_index:end_index + 1]
                answer_text = tokenizer.decode(answer_tokens, skip_special_tokens=True)
                predictions[example_id] = answer_text

    with open(output_file, "w") as f:
        json.dump(predictions, f)
    print(f"Predictions saved to {output_file}")

if __name__ == "__main__":
    train_model(model, train_loader, val_loader, epochs=10)

    save_model(model, tokenizer)

    model, tokenizer = load_model()
    
    generate_predictions(model, val_loader, tokenizer)
