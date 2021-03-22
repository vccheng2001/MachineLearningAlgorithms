import torch
import torch.nn as nn
from torch.autograd import Variable


class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(FlowLSTM, self).__init__()
        # build your model here
        # your input should be of dim (batch_size, seq_len, input_size)
        # your output should be of dim (batch_size, seq_len, input_size) as well
        # since you are predicting velocity of next step given previous one
        
        # feel free to add functions in the class if needed
        
        # input_size == number_features (number variables in time series) == 1
        # input_size == output_size == (bs, 19, 17) 
        # nn.LSTMCell(input_size, hidden_size, bias=True)

        ''' In training set, your input is of dimension (batch_size, 19, 17) 
            and ground truth us of dimension (batch_size, 19, 17)'''

        self.input_size = input_size        # num features in x
        self.hidden_size = hidden_size      # num features in h
        self.num_layers = num_layers        # num layers 
        self.dropout = dropout              # dropout probability 


    # forward pass through LSTM layer
    def forward(self, x):
        '''
        input: x of dim (batch_size, 19, 17)
        '''
        # define your feedforward pass
        # x = torch.randn(batch_size, seq_len, input_size)

        # LSTMCell(num features in x, num features in h, bias)
        lstm = nn.LSTMCell(input_size, 



    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17)
        '''
        # define your feedforward pass
        return NotImplementedError