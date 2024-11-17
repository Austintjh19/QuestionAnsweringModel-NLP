import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, DataCollatorWithPadding
from datasets import load_dataset, Dataset
import numpy as np
from tqdm import tqdm
import json

class RetroReader(nn.Module):
    def __init__(self, model_name="bert-base-uncased", weight_sketchy=0.5, weight_intensive=0.5):
        super(RetroReader, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.answerable_head = nn.Linear(self.encoder.config.hidden_size, 2)
        self.cross_attention = nn.MultiheadAttention(self.encoder.config.hidden_size, num_heads=8)
        self.start_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.end_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.verifier_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.weight_sketchy = weight_sketchy
        self.weight_intensive = weight_intensive

    def forward(self, input_ids, attention_mask, question_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        cls_token = sequence_output[:, 0, :]
        answer_logits = self.answerable_head(cls_token)
        score_null = answer_logits[:, 1] - answer_logits[:, 0]

        question_tokens = sequence_output * question_mask.unsqueeze(-1)
        question_attended, _ = self.cross_attention(
            question_tokens.permute(1, 0, 2), 
            sequence_output.permute(1, 0, 2), 
            sequence_output.permute(1, 0, 2)
        )
        question_attended = question_attended.permute(1, 0, 2)

        start_logits = self.start_head(question_attended).squeeze(-1)
        end_logits = self.end_head(question_attended).squeeze(-1)
        
        verifier_score = self.verifier_head(cls_token)

        score_has = start_logits.max(dim=1).values + end_logits.max(dim=1).values
        final_score = self.weight_sketchy * score_null + self.weight_intensive * score_has + verifier_score.squeeze(-1)
        final_decision = final_score > 0

        return final_decision, start_logits, end_logits, answer_logits, verifier_score

def prepare_training_data():
    dataset = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    def preprocess(examples):
        inputs = tokenizer(
            examples['question'], 
            examples['context'], 
            max_length=384, 
            truncation="only_second", 
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_mapping = inputs.pop("overflow_to_sample_mapping")
        start_positions = []
        end_positions = []

        for i, offsets in enumerate(offset_mapping):
            sample_index = sample_mapping[i]
            answer = examples["answers"][sample_index]
            
            cls_index = inputs["input_ids"][i].index(tokenizer.cls_token_id)
            
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            token_start_index = 0
            token_end_index = 0
            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:
                    token_start_index = idx
                if start < end_char <= end:
                    token_end_index = idx
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
        
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names
    )
    
    return tokenized_dataset

def prepare_prediction_data(dataset):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    def preprocess(examples):
        tokenized_inputs = tokenizer(
            examples['question'],
            examples['context'],
            max_length=384,
            truncation="only_second",
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length"
        )
        
        sample_mapping = tokenized_inputs.pop("overflow_to_sample_mapping")
        
        original_ids = [str(examples["id"][sample_mapping[i]]) for i in range(len(tokenized_inputs["input_ids"]))]
        
        tokenized_inputs["original_id"] = original_ids
        
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    return tokenized_dataset

class CustomDataCollator:
    def __init__(self, tokenizer):
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    def __call__(self, features):
        original_ids = [feature.pop("original_id") for feature in features]
        
        batch = self.data_collator(features)
        
        batch["original_id"] = original_ids
        
        return batch

def train_model_and_save(model, train_dataset, val_dataset, epochs=3, batch_size=8, lr=3e-5, save_path="retro_reader_model.pth"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}/{epochs}")
        
        for batch in train_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            question_mask = (input_ids == tokenizer.sep_token_id).cumsum(dim=1).to(device) == 1

            optimizer.zero_grad()
            final_decision, start_logits, end_logits, _, _ = model(input_ids, attention_mask, question_mask)

            start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
            end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)
            loss = start_loss + end_loss
            total_train_loss += loss.item()
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            train_bar.set_postfix({'train_loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0
        val_bar = tqdm(val_loader, desc=f"Validation Epoch {epoch + 1}/{epochs}")
        
        with torch.no_grad():
            for batch in val_bar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                start_positions = batch["start_positions"].to(device)
                end_positions = batch["end_positions"].to(device)
                question_mask = (input_ids == tokenizer.sep_token_id).cumsum(dim=1).to(device) == 1

                final_decision, start_logits, end_logits, _, _ = model(input_ids, attention_mask, question_mask)

                start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
                end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)
                val_loss = start_loss + end_loss
                total_val_loss += val_loss.item()
                
                val_bar.set_postfix({'val_loss': f'{val_loss.item():.4f}'})

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Average Training Loss: {avg_train_loss:.4f}")
        print(f"Average Validation Loss: {avg_val_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")


def load_model_and_predict(model, val_dataset, model_path="retro_reader_model.pth", output_file="predictions.json", batch_size=8):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = CustomDataCollator(tokenizer)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    predictions = {}
    
    for batch in tqdm(val_loader, desc="Generating predictions"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        question_mask = (input_ids == tokenizer.sep_token_id).cumsum(dim=1).to(device) == 1
        original_ids = batch["original_id"]  # Now this is a list of strings
        
        with torch.no_grad():
            final_decision, start_logits, end_logits, _, _ = model(input_ids, attention_mask, question_mask)
        
        for i in range(input_ids.size(0)):
            if final_decision[i]:
                start_idx = torch.argmax(start_logits[i]).item()
                end_idx = torch.argmax(end_logits[i]).item()
                end_idx = max(start_idx, end_idx)
                answer = tokenizer.decode(input_ids[i][start_idx:end_idx+1], skip_special_tokens=True)
            else:
                answer = ""
            
            question_id = original_ids[i]
            predictions[question_id] = answer
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)
    
    print(f"Predictions saved to {output_file}")
    return predictions

def main(train=True):
    dataset = load_dataset("squad")
    tokenized_dataset = prepare_training_data()
    train_dataset = tokenized_dataset["train"]
    validation_dataset = tokenized_dataset["validation"]
    val_dataset = prepare_prediction_data(dataset["validation"])
    
    model = RetroReader()

    if train:
        print("Starting training...")
        train_model_and_save(
            model=model,
            train_dataset=train_dataset,
            val_dataset=validation_dataset,
            epochs=3,
            batch_size=8,
            lr=3e-5,
            save_path="retro_reader_model.pth"
        )
    else:
        print("Loading model and generating predictions...")
        predictions = load_model_and_predict(
            model=model,
            val_dataset=val_dataset,
            model_path="retro_reader_model.pth",
            output_file="predictions.json",
            batch_size=8
        )

if __name__ == "__main__":
    main(train=True)  # Set to `False` to only test and generate predictions
