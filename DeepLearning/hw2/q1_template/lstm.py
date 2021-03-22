import torch
import torch.nn as nn
from torch.autograd import Variable

''' Fluid flow velocity prediction using LSTM
    Training: predict velocity at next step given velocity at previous step'''

class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(FlowLSTM, self).__init__()

        ''' input dimension:        (batch_size, 19, 17) 
            ground truth dimension: (batch_size, 19, 17)'''

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.input_size = input_size        # num features in x (vector length)
        self.hidden_size = hidden_size      # num LSTM cells per layer 
        self.num_layers = num_layers        # num LSTM/recurrentlayers (vertical)
        self.dropout = dropout              # dropout probability 
        self.seq_len = 0            

        # define LSTM Cell 
        self.lstmCell = nn.LSTMCell(input_size = self.input_size,
                                    hidden_size = self.hidden_size)

        # define Linear layer to get predictions y from h
        self.linear = nn.Linear(self.hidden_size, self.input_size)
    
    # initialize hiddens before train/test: (batch_size, hidden_size)
    def init_hidden_state(self, batch_size):
        return Variable(torch.zeros(batch_size, self.hidden_size),requires_grad=True).to(self.device)

    # forward pass through LSTM layer (training) 
    def forward(self, x):

        '''           
        Many-to-Many
        # input x: [batch_size, seq_len, input_size]
        '''

        (batch_size, self.seq_len, input_size) = x.shape
        # since lstmCell takes in 
        x.transpose(0,1) 
        
        # hidden dims: (batch size, hidden size)
        hx = self.init_hidden_state(batch_size)
        cx = self.init_hidden_state(batch_size)

        output = []
        # at each timestep
        for t in range(self.seq_len):
            # x: batch, input_size 
            hx, cx = self.lstmCell(x[:, t, :], (hx, cx))
            # map output dim from 128 -> 17 
            out = self.linear(hx)
            # append to output array
            output.append(out)
        # stack as tensor  
        output = torch.stack(output, dim = 1)
        return output, (hx, cx)


    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        One-to-Many
        input: x of dim (batch_size, input_size)
        '''
        batch_size, input_size  = x.shape 

        output = []
        out = x      # input to first LSTM cell

        # init hiddens
        hx = self.init_hidden_state(batch_size)
        cx = self.init_hidden_state(batch_size) 

        # at each timestep 
        for t in range(self.seq_len): 
            # lstm cell takes in prev timestep's output as inputs
            (hx, cx) = self.lstmCell(out, (hx, cx))
            # feed output back
            out = self.linear(hx)
            output.append(out)
        # stack as tensor
        output = torch.stack(output, dim = 1)
        return output