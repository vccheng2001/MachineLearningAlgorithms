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

        self.input_size = input_size        # num features in x (vector length)
        self.hidden_size = hidden_size      # num LSTM blocks per layer 
        self.num_layers = num_layers        # num LSTM layers (vertical)
        self.dropout = dropout              # dropout probability 

        # define LSTM layer 
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size, 
                            self.num_layers)


    # forward pass through LSTM layer
    def forward(self, x):
        '''           # sequences, seq length,  # vars in time series 
                        (batch_size, seq_len,  num_features/input_size)
        input: x of dim (batch_size,     19,        17)
        '''
        (self.batch_size, self.seq_len, self.inpu_size) = x.shape
        
        h0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size) # initialize hidden 
        c0 = torch.randn(self.num_layers, self.batch_size, self.hidden_size) # initialize cell
        # concat h0, c0 as hidden
        hidden = (h0, c0)
        output, (hn,cn) = self.lstm(x, hidden)
        print(f"output shape: {output.shape}")
        print(f"hidden: {hidden}")
        


    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17)
        '''
        # define your feedforward pass
        return NotImplementedError