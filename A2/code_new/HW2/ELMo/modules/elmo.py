import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMMP(nn.Module):
    """

    Args:

    """

    def __init__(self, input_size,num_layers , hidden_size, projection_size ):
        super(LSTMMP, self).__init__()
        self.hidden_size = hidden_size # hidden size
        self.num_layers= num_layers
        #You cannot use bi-LSTM
        #for layer_index in range(num_layers):
            
        self.rnn_forward_1=nn.LSTM(input_size,self.hidden_size, num_layers=1,bidirectional=False , batch_first=True )
        self.projection_forward_1 = nn.Linear(hidden_size, projection_size)

        self.rnn_forward_2=nn.LSTM(input_size ,self.hidden_size, num_layers=1,bidirectional=False , batch_first=True )
        self.projection_forward_2 = nn.Linear(hidden_size, projection_size)
        
        self.rnn_backward_1=nn.LSTM(input_size,self.hidden_size, num_layers=1,bidirectional=False , batch_first=True )
        self.projection_backward_1 = nn.Linear(hidden_size, projection_size)
        
        self.rnn_backward_2=nn.LSTM(input_size ,self.hidden_size, num_layers=1,bidirectional=False , batch_first=True )
        self.projection_backward_2 = nn.Linear(hidden_size, projection_size)
        

    def forward(self, x):
        #print(x.shape) #batch_size, seq_len, projection_size=512
        #https://discuss.pytorch.org/t/how-to-use-a-lstm-in-a-reversed-direction/14389/2
        batch_size, max_sent_len, _ = x.shape
        #print(max_sent_len)
        for_indx= torch.arange(max_sent_len-1).long() # 0~max_sent_len-2
        #print(for_indx)
        back_indx= torch.arange(max_sent_len-1, 0, -1).long() # max_sent-1 ~1
        
        for_indx= for_indx.cuda()
        back_indx = back_indx.cuda()
        #print(back_indx)
        #input()
        f_x=  x.index_select(1, for_indx)
        b_x=  x.index_select(1, back_indx)
        #print("f_x: ", f_x)
        #print("b_x: ", b_x)
        #print(f_x.shape) #batch_size, max_sent_len-1, projection_size=512
        #print(b_x.shape) #the same as fx.shape
        #input()
        f_output, (h_n_f, c_n_f) = self.rnn_forward_1(f_x)
        b_output, (h_n_b, c_n_b) =self.rnn_backward_1(b_x)
        #print(f_output.shape) #batch_size, seq_len-1,hidden_size=2048
        #print(b_output.shape) #the same as above
        #For the unpacked case, the directions can be separated using output.view(seq_len, batch, num_directions, hidden_size), with forward and backward being direction 0 and 1 respectively. 
        f_1=self.projection_forward_1(f_output)
        b_1=self.projection_backward_1(b_output)
        #output= output.view(batch_size, max_sent_len, 2,self.hidden_size )
        #print("self.training: ", self.training)
        #output= F.dropout(output, 0.1, self.training)        
        #print(f_1.shape) #batch_size, seq_len-1, 512
        #print(b_1.shape) #the same as above
        f_output, (h_n_f, c_n_f) = self.rnn_forward_2(f_1)
        b_output, (h_n_b, c_n_b) =self.rnn_backward_2(b_1)
        #input()
        #should we concat f_output and b_output? 
        f=self.projection_forward_2(f_output)
        b=self.projection_backward_2(b_output)
        #print(b.size(1))
        #print(f.shape) #batch_size, seq_len-1, 512
        #print(b.shape) #batch_size, seq_len-1, 512
        sz0, sz1, sz2 = f.shape

        reverse_indx = torch.arange(b.size(1)-1, -1, -1).long() #seq_len-2,.., 0
        #print(reverse_indx)
        #input()
        reverse_indx = reverse_indx.cuda()
        
        b= b.index_select(1, reverse_indx)
        b_1= b_1.index_select(1, reverse_indx)
         
        #concat (f_1, f) and (b_1, b) --> shape (2, batch_size, seq_len-1, projeciton_size) 
        #print(f_1.unsqueeze(0).shape) #1, batch_size, seq_len-1 , projection_size
        
        f= torch.cat( (f_1.unsqueeze(0), f.unsqueeze(0)), 0) #2, batch_size, seq_len-1, projection_size
        b= torch.cat((b_1.unsqueeze(0), b.unsqueeze(0)), 0)
        #f[0] is the 1-st layer, f[1] the 2nd layer 
        #print(f.shape) #2, batch_size, seq_len-1, projection_size
        #print(b.shape) #the same
        #input()

        return f, b
