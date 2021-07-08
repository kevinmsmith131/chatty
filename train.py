import json
import numpy as np
import torch
import torch.nn as nn
from model import NeuralNet
from torch.utils.data import Dataset, DataLoader
from nltk_utils import tokenize, stem_and_lower, bag_of_words

with open('intents.json', 'r') as file:
    intents = json.load(file)

all_words = []
tags = []
xy = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    
    for pattern in intent['patterns']:
        token = tokenize(pattern)
        all_words.extend(token)
        xy.append((token, tag))

ommited_chars = ['?', '!', '.', ',']
all_words = [stem_and_lower(word) for word in all_words if word not in ommited_chars]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

X_train = []
y_train = []

for (phrase, tag) in xy:
    bag = bag_of_words(phrase, all_words)
    X_train.append(bag)

    label = tags.index(tag) 
    y_train.append(label)

X_train = np.array(X_train)
y_train = np.array(y_train)

batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(X_train[0])
learning_rate = 0.001
num_epochs = 1000

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples


dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(dtype=torch.long).to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'epoch {epoch + 1} / {num_epochs}, loss={loss.item():.4f}')

print(f'final loss, loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags 
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'finished training. file saved to {FILE}')
