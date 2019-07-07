import torch
import torch.nn as nn

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()

        self.method = method
        self.hidden_size = hidden_size

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
            self.attn2 = nn.Linear(self.hidden_size, self.hidden_size)   
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, opt_output, context_outputs):
        # hidden [batch_size,opt_len, 256], encoder_outputs [batch_size,context_len ,256]
        context_len = context_outputs.size(1) #length of the context
        batch_size = context_outputs.size(0)
        hidden_size=  opt_output.size(2)  #256
        opt_len = opt_output.size(1)
        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size,opt_len, context_len) # B x opt_len  x S
        #attn_energies= torch.zeros(batch_size, context_len, opt_len) #B x context_len x opt_len
        attn_energies = attn_energies.to(torch.device('cuda:0'))
        #print("context_outputs.shape: ", context_ouputs.shape) #torch.Size([10, 39, 256])
        attn_energies= torch.bmm(opt_output, context_outputs.transpose(1,2)) # (batch, opt_len, dim) * (batch, context_len, dim) ->  (batch, opt_len, context_len)
        return nn.functional.softmax(attn_energies.view(-1, context_len), dim=1).view(batch_size,-1, context_len )
#score function will be used when calculating energy
#about squeeze(0): https://pytorch.org/docs/stable/torch.html#torch.squeeze
    def score(self, opt_output, context_output):
        # opt_output [seq_len, 256], context_output [1, 256]
        if self.method == 'dot':
            energy = opt_output.squeeze(0).dot(context_output.squeeze(0))
            return energy

        elif self.method == 'general':
            energy = self.attn(context_output)
            energy = opt_output.squeeze(0).dot(energy.squeeze(0))
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((opt_output, context_output), 1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
            return energy
#How to visualize the attention??
#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class RNNAtt(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, hidden_size,dropout , 
                 similarity='inner_product'):
        super(RNNAtt, self).__init__()
        self.hidden_size = hidden_size # hidden size can be anything~ depends on you
        self.num_layers= 1
        self.rnn_encoder = nn.LSTM(dim_embeddings, self.hidden_size,num_layers=1,dropout=0 ,bidirectional=True, batch_first=True)     
        attn_model= 'general'
        # UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.05 and num_layers=1 --> see what happens
        self.attn = Attn(attn_model ,self.hidden_size*2 )
        #read https://pytorch.org/docs/stable/nn.html?highlight=rnn#torch.nn.RNN
        #we do not need decoder here; we can use the same encoder
        self.rnn_2 = nn.LSTM(8*hidden_size , hidden_size,num_layers=1,dropout=0 ,bidirectional= True, batch_first=True)
        self.W =nn.Bilinear(2*self.hidden_size,2*self.hidden_size, 1)


    def forward(self, context, context_lens, options, option_lens):
        #context = self.rnn_encoder(context)[0].max(1)[0] #max-pooling over RNN outputs; torch.Size([10, 100])
        output, (hn, cn)=self.rnn_encoder(context)
        #print("output.shape: ", output.shape) #torch.Size([10, 27, 256]) batch_size, seq_length, 2*hidden_size
        logits = []
        attention= [] 
        #print("options before mlp: ", options.shape) #torch.Size([10, 5, 50, 200]); evaluation 300d: torch.Size([10, 5, 50, 300])
        for i, option in enumerate(options.transpose(1, 0)):
            #print("option before rnn: ", option.shape) #torch.Size([10, 39, 300])
            #option = self.rnn_encoder(option)[0].max(1)[0]  #max-pooling over RNN outputs
            opt_out, (hn_o, cn_o)=self.rnn_encoder(option)
            #https://pytorch.org/docs/stable/torch.html?highlight=max#torch.max
            #so we will caculate the attention for 100 times
            #I think we should use the last hidden state to represent the whole sentence of the option
            attn_weights= self.attn(opt_out, output)
            #print("")
            attn=  torch.bmm(attn_weights, output) # (B xopt_len x context_len ) x (B x  context_len x256)-->(Bx opt_lenx 256)
            attn_weights= nn.functional.softmax(attn_weights, dim=1 )
            #input()
            big_vector = torch.cat((opt_out, attn, opt_out*attn, opt_out-attn),2 )
            #why we need to concat vectors containing context informaion and vectors containing option information??????
            #input()#so this big vector should store some information of each candidate response
            output_2, (hn_2, cn_2)= self.rnn_2(big_vector)
            #u= hn_2.contiguous().view(hn_2.shape[0], -1) 
            u = output_2.max(1)[0] #max-pooling #u is the vector after second layer of RNN
            v=  opt_out.max(1)[0]
            u= u.unsqueeze(1)
            v= v.unsqueeze(2)

            #v= hn.contiguous().view(hn.shape[0], -1) #use the last hidden state of the opt_out IS WRONG!!! you have to use the last hidden state of the context output
            #logit= self.W(v, u).sum(-1)
            logit = (torch.bmm(u, v).squeeze(2)).sum(-1)
            #W is trainable; how to represent W ????? logit is the score
            #how to compute the score after calculating the attention??????
            logits.append(logit)
            attention.append(attn_weights)
            #print("i= ", i) #i=0~4; evaluation i=0~4
        #print("logits before torch.stack: ",len(logits)) # len(logits)=5; the same as example net(?)
        logits = torch.stack(logits, 1) #Concatenates sequence of tensors along a new dimension. All tensors need to be of the same size.
        return logits, attention
