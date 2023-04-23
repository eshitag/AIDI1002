# Install the necessary libraries
!pip install transformers
!pip install torch
!pip install scikit-learn
# Import the required libraries
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# Load the dataset and split it into train and test sets
newsgroups = fetch_20newsgroups(subset='all')
train_data, test_data, train_labels, test_labels = train_test_split(newsgroups.data, newsgroups.target, test_size=0.9, random_state=42)

# Tokenize the input text
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(train_data, truncation=True, padding=True, max_length=512)
test_encodings = tokenizer(test_data, truncation=True, padding=True, max_length=512)

# Convert the labels to tensors
train_labels = torch.tensor(train_labels)
test_labels = torch.tensor(test_labels)

# Define the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(newsgroups.target_names))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
print("model done")
# Set the optimizer and the training parameters
optimizer = AdamW(model.parameters(), lr=5e-5)
num_epochs = 1
batch_size = 8
count = 0
# Create a PyTorch DataLoader object for the training and test data
train_dataset = torch.utils.data.TensorDataset(torch.tensor(train_encodings['input_ids']), torch.tensor(train_encodings['attention_mask']), train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(torch.tensor(test_encodings['input_ids']), torch.tensor(test_encodings['attention_mask']), test_labels)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
print("object created")
# Train the model
for epoch in range(num_epochs):
    #print("traincount : "+ str(count+1))
    # Training
    model.train()
    for batch in train_loader:
        # Load the data to the device
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        labels = batch[2].to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        # Backward pass
        loss.backward()

        # Update the parameters
        optimizer.step()
    print("evaluation started")
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    count = 0
    with torch.no_grad():
        for batch in test_loader:
            #print("Evalcount : "+ str(count+1))
            # Load the data to the device
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            # Forward pass
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=1)

            # Compute the accuracy
            total += labels.size(0)
            correct += (predictions == labels).sum().item()

    accuracy = correct / total
    print('Epoch:', epoch, 'Accuracy:', accuracy)

