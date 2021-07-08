import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as file:
    intents = json.load(file)

FILE = 'data.pth'
data = torch.load(FILE)

model_state = data['model_state']
input_size = data['input_size']
output_size = data['output_size']
hidden_size = data['hidden_size']
all_words = data['all_words']
tags = data['tags']

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = 'Chatty'
print("I'm here to chat! Type 'quit' (casing does not matter) to exit.")
while True:
    phrase = input('You: ')
    if phrase.lower() == 'quit':