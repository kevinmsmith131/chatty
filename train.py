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

x_train = []
y_train = []

for (phrase, tag) in xy:
    bag = bag_of_words(phrase, all_words)
    x_train.append(bag)

    label = tags.index(tag) 
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)


class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data

    def __len__(self):
        return self.n_samples


batch_size = 8
hidden_size = 8
output_size = len(tags)
input_size = len(all_words)
learning_rate = 0.001
num_epochs = 1000

dataset = ChatDataset()
rain_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)