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
    
        ''' In training set, your input is of dimension (batch_size, 19, 17) 
            and ground truth is of dimension (batch_size, 19, 17)'''

        self.input_size = input_size        # num features in x (vector length)
        self.hidden_size = hidden_size      # num LSTM cells per layer 
        self.num_layers = num_layers        # num LSTM/recurrentlayers (vertical)
        self.dropout = dropout              # dropout probability 

        # define LSTM Cell
        self.lstm = nn.LSTMCell(input_size  = self.input_size,
                                hidden_size = self.hidden_size)

        self.linear = nn.Linear(self.hidden_size, self.input_size)
    

    # forward pass through LSTM layer
    def forward(self, x):
        '''           
        # Size: [batch_size, seq_len, input_size]
        input x: (batch_size,   19,     17)
        '''

        (self.batch_size, self.seq_len, self.input_size) = x.shape
        
        # batch size, hidden size
        hx = torch.randn(self.seq_len, self.hidden_size)
        cx = torch.randn(self.seq_len, self.hidden_size) 

        output = []
        # for each input x[i] in batch
        for i in range(self.batch_size):
            # hidden for batch i 
            hx, cx = self.lstm(x[i], (hx, cx))
            # map output dim from 128 -> 17 
            out = self.linear(hx)
            # append to output array
            output.append(out)
        # convert output to tensor 
        output = torch.stack(output, dim = 0 )
        # print(f" \n output LSTMCell: {output.shape}")
        return output, (hx, cx)


    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17) [ only one x ]
        '''
        (self.batch_size, self.input_size) = x.shape

        # initialize hidden
        hx = torch.randn(self.seq_len, self.hidden_size)
        cx = torch.randn(self.seq_len, self.hidden_size) 

        output = []
        # for each input x[i] in batch
        for i in range(self.batch_size):
            # instead of 16*19x17, now 16x17 -> transform to 19x17
            inp = x[i].unsqueeze(0)
            inp = inp.repeat(self.seq_len,1)
            # hidden for batch i
            hx, cx = self.lstm(inp, (hx, cx))
            # map output dim from 128 -> 17 
            out = self.linear(hx)
            # append to output array
            output.append(out)
        # convert output to tensor 
        output = torch.stack(output, dim = 0 )
        return output