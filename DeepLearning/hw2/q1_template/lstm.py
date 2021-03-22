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
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_size = input_size        # num features in x (vector length)
        self.hidden_size = hidden_size      # num LSTM cells per layer 
        self.num_layers = num_layers        # num LSTM/recurrentlayers (vertical)
        self.dropout = dropout              # dropout probability 
        self.seq_len = 0
        self.hidden = None

        # define LSTM Cell (for test)
        self.lstmCell = nn.LSTMCell(input_size = self.input_size,
                                    hidden_size = self.hidden_size)

        self.linear = nn.Linear(self.hidden_size, self.input_size)
    


    # forward pass through LSTM layer
    def forward(self, x):

        '''           
        # Size: [batch_size, seq_len, input_size]
        input x: (batch_size,   19,     17)
        '''

        (batch_size, self.seq_len, input_size) = x.shape
        x.transpose(0,1) # seq_len, batch_size, input_size 
        
        # batch size, hidden size
        hx = self.init_hidden_state(batch_size)
        cx = self.init_hidden_state(batch_size)

        output = []
        # for each input x[i] in batch
        for t in range(self.seq_len):
            # x: batch, input_size 
            hx, cx = self.lstmCell(x[:, t, :], (hx, cx))
            # map output dim from 128 -> 17 
            out = self.linear(hx)
            # append to output array
            output.append(out)
        # convert output to tensor  
        output = torch.stack(output, dim = 1)
        # save hidden 
        self.hidden =  (hx, cx)
        return output, self.hidden


    def init_hidden_state(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size)).to(self.device)

    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17)
        '''
        batch_size, input_size  = x.shape #
        # hiddens: batch size, hidden size
        output = []
        # for each input x[i] in batch
        out = x
        # from training 
        hx = self.init_hidden_state(batch_size)
        cx = self.init_hidden_state(batch_size) #self.hidden 

        for t in range(self.seq_len): 
            # lstm Cell takes in prev timestep's output as inputs
            (hx, cx) = self.lstmCell(out, (hx, cx))
            # feed output (hx) back
            out = self.linear(hx)
            output.append(out)
        # stack for each timetstep
        output = torch.stack(output, dim = 1)
        return output