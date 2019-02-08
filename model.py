import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch

class LSTMBaseline(nn.Module):
    """A typical LSTM with a linear classification layer and softmax output.

    It's potentially multilayered (num_layers).
    We use a special initialization on the LSTM of setting biases to 1 to not forget previous hidden states"""
    def __init__(self, input_size, batch_size, hidden_dim=32, num_layers=1, ):
        super(LSTMBaseline, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.input_size = input_size
        self.batch_size = batch_size

        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True)

        self.linear_layer = nn.Linear(hidden_dim, 10)
        self.hidden = self.init_hidden()

        # A better initialization scheme: don't forget!
        for names in self.lstm._all_weights:
            for name in filter(lambda n: "bias" in n, names):
                bias = getattr(self.lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data[start:end].fill_(1.)

    def init_hidden(self):
        return (Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda(),
                Variable(torch.zeros(self.num_layers, self.batch_size, self.hidden_dim)).cuda())

    def forward(self, x):
        x, self.hidden = self.lstm(x, self.hidden)
        x = x[:, -1, :]
        x = self.linear_layer(x.view(-1, self.hidden_dim))
        return F.log_softmax(x, dim=1)
