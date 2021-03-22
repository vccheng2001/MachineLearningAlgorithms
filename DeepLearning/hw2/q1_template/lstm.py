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
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_size = input_size        # num features in x (vector length)
        self.hidden_size = hidden_size      # num LSTM cells per layer 
        self.num_layers = num_layers        # num LSTM/recurrentlayers (vertical)
        self.dropout = dropout              # dropout probability 
        self.seq_len = None # init

        # define LSTM Cell
        self.lstm = nn.LSTM(input_size  = self.input_size,
                            hidden_size = self.hidden_size,
                            num_layers = self.num_layers,
                            dropout = self.dropout)

        self.linear = nn.Linear(self.hidden_size, self.input_size)
    

    def init_hidden_state(self, seq_len):
        return Variable(torch.zeros(self.num_layers, seq_len, self.hidden_size))

    # forward pass through LSTM layer
    def forward(self, x):

        '''           
        # Size: [batch_size, seq_len, input_size]
        input x: (batch_size,   19,     17)
        '''

        (batch_size, seq_len, input_size) = x.shape
        
        # batch size, hidden size
        h0 = self.init_hidden_state(seq_len)
        c0 = self.init_hidden_state(seq_len)

        out, (hn,cn) =  self.lstm(x, (h0, c0))
        out          =  self.linear(out) 
        return out, (hn, cn)


    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17) [ only one x ]
        '''

        (batch_size, input_size) = x.shape

        # initialize hidden
        hx = torch.randn(self.seq_len, self.hidden_size)
        cx = torch.randn(self.seq_len, self.hidden_size) 

        output = []
        # for each input x[i] in batch
        for i in range(batch_size):
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