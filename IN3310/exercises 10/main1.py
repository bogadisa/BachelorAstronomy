import torch
import torch.nn as nn
import csv
import string
import random

all_letters = string.ascii_letters + "0123456789 .,:!?’[]()/+-="
n_letters = len(all_letters) + 1

def get_data():
    category_lines = {}
    all_categories = ["st"]
    category_lines["st"] = []

    filterwords=["NEXTEPISODE"]
    with open("./star_trek_transcripts_all_episodes.csv", newline="") as csvfile:
        reader = csv.reader(csvfile, delimiter=",", quotechar='"')
        for row in reader:
            for el in row:
                if (el not in filterwords) and (len(el)>1):
                    # print(el)
                    v=el.strip().replace(";","").replace("\"","")
                    category_lines["st"].append(v)
    n_categories = len(all_categories)
    print(len(all_categories), len(category_lines["st"]))
    print("done")
    return category_lines, all_categories


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


# One-hot vector for category
def categoryTensor(category):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li in range(len(line)):
        letter = line[li]
        tensor[li][0][all_letters.find(letter)] = 1
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes)


# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample():
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category)
    input_line_tensor = inputTensor(line)
    target_line_tensor = targetTensor(line)
    return category_tensor, input_line_tensor, target_line_tensor


criterion = nn.NLLLoss()

learning_rate = 0.0005

def train(category_tensor, input_line_tensor, target_line_tensor):
    print(input_line_tensor)
    target_line_tensor.unsqueeze_(-1)
    # print(lstm.hidden_size)
    # quit()
    hidden = torch.zeros(1, lstm.hidden_size)

    lstm.zero_grad()

    loss = 0

    # print(input_line_tensor)
    for i in range(input_line_tensor.size(0)):
        # output = lstm(input_line_tensor[i])
        print(lstm(input_line_tensor[i]))
        quit()
        l = criterion(output, target_line_tensor[i])
        loss += l

    loss.backward()

    for p in lstm.parameters():
        p.data.add_(p.grad.data, alpha=-learning_rate)

    return output, loss.item() / input_line_tensor.size(0)

import time
import math

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == "__main__":
    category_lines, all_categories = get_data()
    n_categories = len(all_categories)
    # rnn = RNN(n_letters, 128, n_letters)
    lstm = nn.LSTM(input_size=1, hidden_size = 100, num_layers=2, dropout=0.1)

    n_iters = 100000
    print_every = 5000
    plot_every = 500
    all_losses = []
    total_loss = 0 # Reset every plot_every iters

    start = time.time()

    for iter in range(1, n_iters + 1):
        output, loss = train(*randomTrainingExample())
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' % (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0