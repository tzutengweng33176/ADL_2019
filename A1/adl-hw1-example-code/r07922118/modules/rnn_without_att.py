import torch
import torch.nn as nn


class RNN(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, hidden_size, 
                 similarity='inner_product'):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size # hidden size can be anything~ depends on you
        self.num_layers= 1
        self.rnn = nn.LSTM(dim_embeddings, hidden_size, self.num_layers ,bidirectional=True, batch_first=True)
        #self.rnn = nn.GRU(dim_embeddings, hidden_size, self.num_layers ,bidirectional=True, batch_first=True)
        # UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.05 and num_layers=1 --> see what happens
        self.W =nn.Bilinear(2*self.hidden_size,2*self.hidden_size, 1)
        #https://pytorch.org/docs/stable/nn.html?highlight=nn%20bilinear#torch.nn.Bilinear
        #read https://pytorch.org/docs/stable/nn.html?highlight=rnn#torch.nn.RNN

    def forward(self, context, context_lens, options, option_lens):
        #print("context before rnn:", context.shape) #  torch.Size([10, 52, 300])
        #print("context after rnn:", self.rnn(context)[0].shape) # output; torch.Size([10, 33, 100]; bidirect :torch.Size([10, 28, 256])
        #print("context after rnn:", len(self.rnn(context)[1])) # (h_n, c_n); a tuple of length 2;a tuple of 2 tensors 
        #print("context after rnn:", self.rnn(context)[1][0].shape) # torch.Size([1, 43, 100]) ; bidirect: torch.Size([2, 28, 128])
        #print("context after rnn:", self.rnn(context)[1][1].shape) # torch.Size([1, 43, 100]); bidirect: torch.Size([2, 28, 128])
        #print("context after rnn.max:", self.rnn(context).max(1)[0].shape) # torch.Size([10, 76, 200]); evaluation 300d:  torch.Size([10, 30, 300])
        #you have to concatenate all the contexts
        #try last hidden state or average pooling 
        #context = self.rnn(context)[0].max(1)[0] #max-pooling over RNN outputs; torch.Size([10, 100])
        context = self.rnn(context)[0].max(1)[0] #try last hidden state
        #context = self.rnn(context)[1][0] #performance did not increase using last hidden state
        #print("context after rnn.max: ", context.shape)#torch.Size([10, 256]);  evalutaion 300d: torch.Size([10, 256]) ; bidirect: torch.Size([10, 256] ); torch.Size([2, 10, 128])
        #input()
        #print("context[0].shape: ", context[0].shape)
        #context=torch.cat((context[0], context[1]), dim=1)
        #print("context.shape : ", context.shape)
        #input()
        #context=context.view(context.shape[0], -1)
        #print("context.shape after view: ", context.shape)
        #is it because I did not use pack_padded_sequence??????
        logits = []
        #print("options before mlp: ", options.shape) #torch.Size([10, 5, 50, 200]); evaluation 300d: torch.Size([10, 5, 50, 300])
        for i, option in enumerate(options.transpose(1, 0)):
            #print("option before rnn: ", option.shape) #torch.Size([10, 39, 300])
            #option = self.rnn(option)[0].max(1)[0]  #max-pooling over RNN outputs
            option = self.rnn(option)[0].max(1)[0]  #try last hidden state
            #option = self.rnn(option)[1][0]
            #https://pytorch.org/docs/stable/torch.html?highlight=max#torch.max
            #print("option after rnn: ", option.shape) # torch.Size([10, 100]);bidirect:  torch.Size([10, 256]) ; torch.Size([2, 10, 128])
            #option=torch.cat((option[0], option[1]), dim=1)
            #print("option.shape after transpose: ", option.shape)
            #option= option.view(option.shape[0], -1)
            #input()
            #logit =self.W(option,context)  #W is trainable; how to represent W ????? logit is the score
            logit =self.W(context, option).sum(-1)  #W is trainable; how to represent W ????? logit is the score
            #W is a linear layer....so I think we should define an nn.Linear in the __init__
            #sum(-1): axis may be negative, in which case it counts from the last to the first axis.
            #ex. if an array has dimension 0, 1, 2; sum(-1) == sum(2); sum(-2)==sum(1)
            #print("logit: ", logit)
            #print("logit.sum(-1): ",logit.sum(-1) )
            #print("logit.shape: ", logit.shape)  # torch.Size([10])
            #input()
            logits.append(logit)
            #print("i= ", i) #i=0~4; evaluation i=0~4
        #print("logits before torch.stack: ",len(logits)) # len(logits)=5; the same as example net(?)
        logits = torch.stack(logits, 1) #Concatenates sequence of tensors along a new dimension. All tensors need to be of the same size.
        #print("logits after torch.stack: ", logits.shape) #torch.Size([10, 5]) ; the same as example net
        #input()
        return logits
