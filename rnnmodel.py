import tensorflow
import torch
import torch.nn as nn
import re

### Code block to def our data in this case it will be text.###



### prepocess our data ###

def preprocessText(text):
    ### This will make our text all lowercast, remove punctuation and non-alphanumeric characters, and we split the data to individual tokens ###
    text = text.lower()
    text = re.sub(r'\d+','',text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^\w\s]', '', text)
    return text.split()

### According to some googling a recurrent neural network (RNN) should be enough for our language model. Building a RNN using pyTorch ###

class rnnModel(torchNN.module): ### torch.nn is PyTorch's base class for all neural network modules. ###
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(rnnModel, self).__init__()
        ### Converts tokens indices into dense vectors ###
        self.embed = nn.Embedding(vocab_size, embed_size)
        ### Processes the sequences of embedded tokens using the specified number of hidden features (hidden_size) and layers (num_layers). ###
        self.rnn = nn.RNN(embed_size, hidden_size, num_layers, batch_first=True)
        ### maps the RNN output features to the vocabulary size, essentially predicting the probability distribution of the next token in the sequence for each position. ###
        self.linear = nn.Linear(hidden_size, vocab_size)

    ### The method where the input tensor (x) goes through the layers of the neural network
    def forward(self, x, h):
        ### x is the input tensor containing token indices. ###
        x = self.embed(x)
        ### h is the initial hidden state of the RNN
        out, h = self.rnn(x, h)
        ### this should be what predicts the next token in the sequence, ??maybe?? lol ###
        out = self.linear(out.reshape(out.size(0)*out.size(1), out.size(2)))
        return out, h

### train our model ###
def trainModel(model, data, epochs, lr):
    ### Same optimizer i used for image generation ###
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    ### Measures the model's prediction error ###
    lossMeasure = nn.CrossEntropyLoss()
    ### Explicitly set the model to training mode ###
    model.train()
    ### Training loop ###
    for epoch in range(epochs):
        ### the hidden state from the previous epoch can be used in the current batch of training. This stops that ###
        hidden = None
        for x, y in data:
            ### reset gradients to zero, google told me  by default, gradients accumulate in PyTorch to handle scenarios like recurrent neural networks. We might not want this depending how complex our data is. ###
            optimizer.zero_grad()
            ### passes the input x and the initial/previous hidden state hidden to the model, receiving the predicted outputs and the next hidden state. ###
            outputs, hidden = model(x, hidden.detach() if hidden is not None else None)
            ### calculates the loss using the lossMeasure by comparing the outputs of the model against the target y###
            loss = lossMeasure(outputs, y.view(-1))
            ###  calculate the gradients of the loss with respect to the model parameters. ###
            loss.backward()
            ### updates the model parameter ###
            optimizer.step()

### Use our model to generate text, We should be saving a file of our models progress then loading that in for text generation. This is just to get some code down for it. ###
def generateText(model, seed_text, num_words):
    model.eval()
    text = [seed_text]
    for _ in range(num_words):
        x = torch.tensor([text[-1]])
        output, _ = model(x, None)
        _, predicted = torch.max(output, 1)
        text.append(predicted.item())
    return ' '.join(map(str, text))
