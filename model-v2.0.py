import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AdamW, get_linear_schedule_with_warmup, DataCollatorWithPadding
from datasets import load_dataset
import numpy as np
from tqdm import tqdm

# Load and prepare the dataset
def prepare_data():
    # Load the SQuAD dataset
    dataset = load_dataset("squad")
    # Initialize the tokenizer for BERT model
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    # Initialize a list to store IDs of each example
    ids = []

    # Define a preprocessing function to tokenize the examples and prepare labels
    def preprocess(examples):
        # Tokenize the question and context text with specific parameters
        inputs = tokenizer(
            examples['question'], 
            examples['context'], 
            max_length=384,                 # Limit each input to a maximum of 384 tokens
            truncation="only_second",       # Only truncate the context (second sequence)
            return_overflowing_tokens=True, # Enable handling of long contexts that overflow max_length
            return_offsets_mapping=True,    # Return the mapping of characters to tokens for locating answers
            padding="max_length"            # Pad each sequence to the maximum length
        )
        
        # Retrieve necessary mappings from tokenized output
        offset_mapping = inputs.pop("offset_mapping")            # Maps tokens back to their character positions in the context
        sample_mapping = inputs.pop("overflow_to_sample_mapping") # Maps each tokenized chunk to its original example
        start_positions = [] # List to store start positions of answers in tokens
        end_positions = []   # List to store end positions of answers in tokens

        # Process each overflowed example (chunked context) separately
        for i, offsets in enumerate(offset_mapping):
            # Get the index of the original example to which this chunk belongs
            sample_index = sample_mapping[i]
            # Get the answer for this example (SQuAD has guaranteed answers)
            answer = examples["answers"][sample_index]
            # Store the example ID
            ids.append(examples["id"][sample_index])
            
            # Get the CLS token index to be used if no answer is found (though SQuAD 1.1 guarantees answers)
            cls_index = inputs["input_ids"][i].index(tokenizer.cls_token_id)
            
            # Determine character start and end positions of the answer in the original context
            start_char = answer["answer_start"][0]
            end_char = start_char + len(answer["text"][0])

            # Initialize token indices for the start and end of the answer span
            token_start_index = 0
            token_end_index = 0
            # Iterate over the token offset mappings to find token-level positions for the answer span
            for idx, (start, end) in enumerate(offsets):
                if start <= start_char < end:  # Check if start_char falls within this token
                    token_start_index = idx
                if start < end_char <= end:    # Check if end_char falls within this token
                    token_end_index = idx
            # Append the calculated start and end token indices to their respective lists
            start_positions.append(token_start_index)
            end_positions.append(token_end_index)
        
        # Append calculated start and end positions to the inputs dictionary
        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    # Apply the preprocessing function to the dataset, removing columns we don’t need
    tokenized_dataset = dataset.map(preprocess, batched=True, remove_columns=dataset["train"].column_names)
    
    # Return the processed dataset and the list of example IDs
    return tokenized_dataset, ids

# RetroReader model as defined earlier (replace with your specific implementation if needed)
class RetroReader(nn.Module):
    def __init__(self, model_name="bert-base-uncased", weight_sketchy=0.5, weight_intensive=0.5):
        # Initialize the RetroReader class
        super(RetroReader, self).__init__()
        
        # Load a pre-trained BERT-based model as the encoder
        self.encoder = AutoModel.from_pretrained(model_name)
        
        # Define a linear layer to predict answerability from the CLS token's hidden state
        self.answerable_head = nn.Linear(self.encoder.config.hidden_size, 2)
        
        # Define a multi-head attention layer to apply cross-attention over context and question
        self.cross_attention = nn.MultiheadAttention(self.encoder.config.hidden_size, num_heads=8)
        
        # Define linear layers to predict start and end positions for the answer span
        self.start_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.end_head = nn.Linear(self.encoder.config.hidden_size, 1)
        
        # Define a verifier head to check answer confidence based on CLS token
        self.verifier_head = nn.Linear(self.encoder.config.hidden_size, 1)
        
        # Set weights for combining sketchy (null) and intensive (answerable) scores
        self.weight_sketchy = weight_sketchy
        self.weight_intensive = weight_intensive

    def forward(self, input_ids, attention_mask, question_mask):
        # Pass inputs through the encoder model to get hidden states
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Extract the last hidden layer's outputs
        sequence_output = outputs.last_hidden_state
        
        # Extract the CLS token's representation (for answerability check)
        cls_token = sequence_output[:, 0, :]
        
        # Pass CLS token through answerable head to determine answerability (logits)
        answer_logits = self.answerable_head(cls_token)
        
        # Compute score for null (no answer) prediction by subtracting answer and no-answer logits
        score_null = answer_logits[:, 1] - answer_logits[:, 0]

        # Mask out non-question tokens in sequence to focus attention on the question tokens
        question_tokens = sequence_output * question_mask.unsqueeze(-1)
        
        # Apply cross-attention on the question tokens and sequence output to enhance question-context interaction
        question_attended, _ = self.cross_attention(
            question_tokens.permute(1, 0, 2), 
            sequence_output.permute(1, 0, 2), 
            sequence_output.permute(1, 0, 2)
        )
        
        # Rearrange back to batch-first format after attention operation
        question_attended = question_attended.permute(1, 0, 2)

        # Compute start and end logits for the answer span prediction
        start_logits = self.start_head(question_attended).squeeze(-1)
        end_logits = self.end_head(question_attended).squeeze(-1)
        
        # Pass CLS token through verifier head to get an additional confidence score
        verifier_score = self.verifier_head(cls_token)

        # Calculate score for answer span (highest scores for start and end positions)
        score_has = start_logits.max(dim=1).values + end_logits.max(dim=1).values
        
        # Combine sketchy and intensive scores with verifier score to make a final decision
        final_score = self.weight_sketchy * score_null + self.weight_intensive * score_has + verifier_score.squeeze(-1)
        
        # Decide if answer exists (True if final score > 0)
        final_decision = final_score > 0

        # Return decision (True/False), start logits, end logits, answerability logits, and verifier score
        return final_decision, start_logits, end_logits, answer_logits, verifier_score


# Training function
def train_model(model, train_dataset, val_dataset, epochs=3, batch_size=8, lr=3e-5):
    # Initialize the tokenizer and data collator for padding
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create a DataLoader for the training dataset
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=data_collator)
    
    # Set up the optimizer (AdamW) and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=lr)
    num_training_steps = epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

    # Check for GPU availability and move model to the appropriate device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set the model to training mode
    model.train()

    # Loop through each epoch
    for epoch in range(epochs):
        # Loop through each batch in the training DataLoader
        for batch in tqdm(train_loader):
            # Move batch inputs to the appropriate device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)
            
            # Calculate the question mask to separate question tokens from context
            # This identifies tokens that are part of the question using the SEP token
            question_mask = (input_ids == tokenizer.sep_token_id).cumsum(dim=1).to(device) == 1

            # Zero out the gradients before backpropagation
            optimizer.zero_grad()
            
            # Forward pass through the model
            final_decision, start_logits, end_logits, _, _ = model(input_ids, attention_mask, question_mask)

            # Calculate start and end loss using cross-entropy (for answer span prediction)
            start_loss = nn.CrossEntropyLoss()(start_logits, start_positions)
            end_loss = nn.CrossEntropyLoss()(end_logits, end_positions)
            
            # Total loss is the sum of start and end position losses
            loss = start_loss + end_loss
            
            # Backward pass to compute gradients
            loss.backward()
            
            # Update the model parameters with the optimizer
            optimizer.step()
            
            # Update the learning rate using the scheduler
            scheduler.step()
        
        # Print the loss after each epoch
        print(f"Epoch {epoch + 1}/{epochs} completed with loss {loss.item()}")


# Prediction function for SQuAD format
def predict(model, val_dataset, batch_size=8):
    # Initialize the tokenizer and data collator for padding
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Create a DataLoader for the validation dataset
    val_loader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=data_collator)
    
    # Check for GPU availability and move model to the appropriate device (GPU if available, else CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Set the model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Dictionary to store predictions as question ID to answer mapping
    predictions = []
    
    # Loop through each batch in the validation DataLoader
    for batch in tqdm(val_loader):
        # Move batch inputs to the appropriate device
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        
        # Calculate the question mask to identify question tokens
        question_mask = (input_ids == tokenizer.sep_token_id).cumsum(dim=1).to(device) == 1
        
        # Disable gradient calculation for inference to save memory and computation
        with torch.no_grad():
            # Forward pass through the model
            final_decision, start_logits, end_logits, _, _ = model(input_ids, attention_mask, question_mask)
        
        # Process predictions for each sample in the batch
        for i in range(input_ids.size(0)):
            # Check if the model predicts the answer as answerable
            if final_decision[i]:
                # Get the start and end token indices for the answer span
                start_idx = torch.argmax(start_logits[i]).item()
                end_idx = torch.argmax(end_logits[i]).item()
                
                # Decode the answer from token IDs back to a string
                answer = tokenizer.decode(input_ids[i][start_idx:end_idx+1], skip_special_tokens=True)
            else:
                # If not answerable, set answer as an empty string
                answer = ""

            # Append the answer (or empty string) to the predictions list
            predictions.append(answer)
    
    # Return the list of predictions
    return predictions


# Run training and prediction
tokenized_dataset, ids = prepare_data()
train_dataset = tokenized_dataset["train"].select(range(500))
val_dataset = tokenized_dataset["validation"]

model = RetroReader()
train_model(model, train_dataset, val_dataset, epochs=3, batch_size=8, lr=3e-5)
predictions = predict(model, val_dataset)


squad_predictions = {id: pred for id, pred in zip(ids, predictions)}

import json
with open("predictions.json", "w") as f:
    json.dump(squad_predictions, f)

import json
with open("ids.json", "w") as f:
    json.dump(ids, f)
