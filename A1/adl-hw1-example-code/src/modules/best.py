import torch
import torch.nn as nn
#About attention:
#https://zhuanlan.zhihu.com/p/31547842

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
        #print("opt_output.shape: ", opt_output.shape) #torch.Size([10, 50, 256])
        #print("opt_output[0, :].shape ",opt_output[0, :].shape ) #the 0-th batch ; torch.Size([50, 256])
        #print("context_outputs[0, 1]: ", context_outputs[0, 1].shape) #0-th row and 1-st column; 0-th batch and 1st word; torch.Size([256])
        #more about numpy indexing--> https://docs.scipy.org/doc/numpy/user/basics.indexing.html#basics-indexing
        #ans_vec = opt_output.contiguous().view(-1,hidden_size ) # (b*opt_len, 256)
        #print("ans_vec.shape: ", ans_vec.shape) #torch.Size([256, 500])
        #mul_ans= torch.t(self.attn(context_outputs)) # (b*h) -->(h*h)x (h*b)-->h*b
        #print("mul_ans.shape: ", mul_ans.shape )# (256, batch)
        #mul_ans= mul_ans.repeat(1, opt_len)  #(256, b*opt_len)
        #ans_vec= torch.t(self.attn2(ans_vec)) ##(256, b*opt_len)  
        #print("ans_vec.shape: ", ans_vec.shape)  #(256, b*opt_len)
        #att_m =mul_ans + ans_vec  ##(256, b*opt_len)
        #rint("att_m.shape: ", att_m.shape) #(256, b*opt_len)
        #att_m= torch.tanh(att_m) #(256, b*opt_len)
        #rint("att_m.shape after tanh: ", att_m.shape) #(256, b*opt_len)
        #print("mul_ans.shape: ", mul_ans.shape)# (256, b*opt_len)
        #nput()
        #s_aq = self.v.mm(att_m) # 1, b*opt
        #s_aq= s_aq.squeeze(0) 
        #print("s_aq.shape: ", s_aq.shape)
        #attn_energies = s_aq.view(-1, opt_len) # (b, opt)
        #revised from https://github.com/divishdayal/Attentive_LSTM_question-answering/blob/master/model.py  
        #print("s_aq.shape: ", attn_energies.shape) # torch.Size([10, opt])
        #input()
        #https://arxiv.org/pdf/1511.04108.pdf
        #context_outputs= self.attn(context_outputs)
        #opt_output= self.attn2(opt_output)
        #print(context_outputs.shape) #(batch, 256)
        #print(opt_output.shape)#torch.Size([batch_size, 44, 256])
        #input()
        #print(context_outputs.unsqueeze(2).shape) #(batch, 256, 1)
        #attn_energies= nn.functional.tanh(context_outputs.unsqueeze(1)+opt_output)
        #print(attn_energies.shape) #torch.Size([10, 39, 256])
        #print(self.v.shape) #torch.Size([1, 256])
        
        #attn_energies=torch.mm(attn_energies.squeeze(0), self.v.transpose(0, 1))
        #attn_energies= torch.bmm(opt_output, context_outputs.unsqueeze(2)).squeeze(2)
        attn_energies= torch.bmm(opt_output, context_outputs.transpose(1,2)) # (batch, opt_len, dim) * (batch, context_len, dim) ->  (batch, opt_len, context_len)
        #attn_energies= torch.bmm(context_outputs, opt_output.transpose(1,2)) #(batch, context_len, dim) *(batch, dim, opt_len)-->(batch, context_len, opt_len)
        #we use the dot method to calculate attention
        #print("attn_energy.shape: ", attn_energies.shape) #(batch, out_len, max_len); (batch, opt_len)
        #print("attn_energy.view(-1, max_len): ", attn_energy.view(-1, max_len).shape)# (batch*out_len, max_len)
        
        #print("shape after softmax: ", nn.functional.softmax(attn_energies.view(-1, max_len), dim=1).shape) #torch.Size([batch*out_len, max_len])
        #input()
                #for every word in context, we calculate the attention energy with each option
                #hidden[:, b] means take the b-th column of all rows 
        #print("attn_energies.shape: ", attn_energies.shape) #torch.Size([10, 32])
        #print("nn.functional.softmax(attn_energies, dim=1): ", nn.functional.softmax(attn_energies, dim=1).shape) #torch.Size([10, 32])
        # Normalize energies to weights in range 0 to 1, resize to  B X opt_len X context_len; B X context_len X opt_len
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
            #print("energy.shape: ", energy.shape) #torch.Size([1, 256])
            #print("energy.squeeze(0): ", energy.squeeze(0).shape) #torch.Size([256])
            #print("opt_output.squeeze(0).shape: ", opt_output.squeeze(0).shape) #torch.Size([256]) after using the last hidden state of option
            #energy=energy.mm(opt_output.transpose(1, 0)).squeeze(0)
            #print("energy.shape: ", energy.shape) #torch.Size([1, 50])
            #print("energy.squeeze(0): ",energy.squeeze(0).shape ) #torch.Size([35])
            energy = opt_output.squeeze(0).dot(energy.squeeze(0))
            #print("energy: ", energy)# torch.Size([])
            #print("energy.shape: ", energy.shape)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((opt_output, context_output), 1))
            energy = self.v.squeeze(0).dot(energy.squeeze(0))
            return energy
#How to visualize the attention??
#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
class BestRNNAtt(torch.nn.Module):
    """

    Args:

    """

    def __init__(self, dim_embeddings, hidden_size,dropout , 
                 similarity='inner_product'):
        super(BestRNNAtt, self).__init__()
        self.hidden_size = hidden_size # hidden size can be anything~ depends on you
        self.num_layers= 2
        self.rnn_encoder = nn.LSTM(dim_embeddings, self.hidden_size,num_layers=2,dropout=0.2 ,bidirectional=True, batch_first=True)     
        attn_model= 'general'
        # UserWarning: dropout option adds dropout after all but last recurrent layer, so non-zero dropout expects num_layers greater than 1, but got dropout=0.05 and num_layers=1 --> see what happens
        self.attn = Attn(attn_model ,self.hidden_size*2 )
        #read https://pytorch.org/docs/stable/nn.html?highlight=rnn#torch.nn.RNN
        #we do not need decoder here; we can use the same encoder
        self.rnn_2 = nn.LSTM(8*hidden_size , hidden_size,num_layers=2,dropout=0.2 ,bidirectional= True, batch_first=True)
        self.W =nn.Bilinear(2*self.hidden_size,2*self.hidden_size, 1)


    def forward(self, context, context_lens, options, option_lens):
        #print("context before rnn:", context.shape) #  torch.Size([10, 29, 300]) ; batch size, seq_length, embd_size
        #print("context_lens: ", context_lens) #[5, 10, 25, 21, 7, 29, 5, 3, 12, 10]; not sorted by decreasing order
        #print("options_lens: ", option_lens) #[[14, 9, 3, 22, 8], [5, 1, 16, 2, 2], [9, 1, 12, 3, 10], [28, 7, 5, 7, 1], [14, 13, 12, 19, 14], [18, 6, 15, 5, 23], [10, 7, 1, 4, 1], [12, 9, 3, 4, 2], [1, 25, 5, 26, 2], [9, 5, 7, 2, 5]] 
        #context = self.rnn_encoder(context)[0].max(1)[0] #max-pooling over RNN outputs; torch.Size([10, 100])
        output, (hn, cn)=self.rnn_encoder(context)
        #print("hn.shape: ", hn.shape) #torch.Size([2, 10, 128])
        #print("hn: ", hn)
        #hn= hn.transpose(0,1)
        #print("hn after transpose: ", hn)
        #print("output[-1]: ", output[-1].shape) # (seq_len, 2*hidden_size)
        #output= output.max(1)[0] #torch.Size([batch_size, 256])
        #input()
        #print("output.shape: ", output.shape) #torch.Size([10, 27, 256]) batch_size, seq_length, 2*hidden_size
        logits = []
        #input()
        #print("options before mlp: ", options.shape) #torch.Size([10, 5, 50, 200]); evaluation 300d: torch.Size([10, 5, 50, 300])
        for i, option in enumerate(options.transpose(1, 0)):
            #print("option before rnn: ", option.shape) #torch.Size([10, 39, 300])
            #option = self.rnn_encoder(option)[0].max(1)[0]  #max-pooling over RNN outputs
            opt_out, (hn_o, cn_o)=self.rnn_encoder(option)
            #https://pytorch.org/docs/stable/torch.html?highlight=max#torch.max
            #print("opt_out.shape: ",opt_out.shape)#torch.Size([10, 50, 256])
            #print("hn_o.shape: ", hn_o.shape) # torch.Size([2, 10, 128])
            #print("hn_o.shape after view: ", hn_o.view(hn_o.shape[1], hn_o.shape[0]*hn_o.shape[2]).shape) # torch.Size([10, 256])
            #print("option after rnn: ", option.shape) # torch.Size([10, 100])
            #input()
            #we can calculate the attention from each opt_output(YES) and context_output
            #there are many ways to caculate attention
            #so we will caculate the attention for 100 times
            #I think we should use the last hidden state to represent the whole sentence of the option
            attn= self.attn(opt_out, output)
            #print("attn: ", attn)
            #print("attn.shape: ", attn.shape) #torch.Size([10, 22, 1]) #  B X opt_len X context_len# Bx opt_len x1
            #input()
            #attn= torch.bmm(attn, opt_out) #weighted sum of the context outputs #b x1 x256
            attn=  torch.bmm(attn, output) # (B xopt_len x context_len ) x (B x  context_len x256)-->(Bx opt_lenx 256)
            #attn= attn.squeeze(1)
            #so this attn has some information about the context
            #print("attn.shape: ", attn.shape)# B X opt_len X 256 ; Bx context_len X 256
            #attn= (attn.max(1)[0]).unsqueeze(1)
            #print("attn.shape: ", attn.shape)
            #attn=attn.repeat(1,opt_out.shape[1] , 1)
            #print("attn.shape: ", attn.shape)  #torch.Size([10,opt_len, 256])
            
            #print("output*attn.shape: ", (opt_out*attn).shape) #torch.Size([10, 27, 256]) #B x opt_len X 256
            #print("(output-attn).shape: ", (opt_out-attn).shape) #torch.Size([10, 27, 256]) #B X opt_len X 256
            #big_vector = attn #if you don't concat with opt_out the performance will be bad
            #print("attn.shape: ", attn.shape)  #torch.Size([10,opt_len, 256])
            
            #opt_out = opt_out.max(1)[0] #max pooling
            #print("opt_out.shape: ", opt_out.shape)#b x opt_len x256
            #input()
            big_vector = torch.cat((opt_out, attn, opt_out*attn, opt_out-attn),2 )
            #why we need to concat vectors containing context informaion and vectors containing option information??????
            #print("big_vector.shape: ", big_vector.shape) #B X opt_len X 8*hidden_size
            #input()#so this big vector should store some information of each candidate response
            output_2, (hn_2, cn_2)= self.rnn_2(big_vector)
            #print("hn_2.shape: ", hn_2.shape) #torch.Size([2, 20, 128])
            #print("output_2.shape: ", output_2.shape) #torch.Size([20, 50, 256])
            #print( )
            #hn_2=hn_2.transpose(0, 1)
            #print("hn_2.shape: ", hn_2.shape)
            #u= hn_2.contiguous().view(hn_2.shape[0], -1) 
            u = output_2.max(1)[0] #max-pooling #u is the vector after second layer of RNN
            v=  output.max(1)[0]
            u= u.unsqueeze(1)
            v= v.unsqueeze(2)

            #v= hn.contiguous().view(hn.shape[0], -1) #use the last hidden state of the opt_out IS WRONG!!! you have to use the last hidden state of the context output
            #max-pooling over opt-output
            #print("u.shape: ", u.shape) #torch.Size([10, 256])
            #print("v.shape: ", v.shape) # B x 2*hidden_size
            #print("what.shape: ", what[0].shape) #torch.Size([10, 49, 256])
            #input()
            #maybe the question you want to ask is why can attention be calculated like this???? Is there any theory to support this kind of calculation?
            #and then we can concat the attention vector a and the context output r like [r;a;r*a;r-a](see slide 41)
            #then put this big vector into the RNN to get the final output
            #we see this final output(maybe we can sum it) as the score of this option
            #print("self.W(u, v).shape: ", self.W(u, v).shape) #torch.Size([10, 1])
            #input()
            #cos =  nn.CosineSimilarity(dim=1, eps=1e-6)  #poor performance
            #logit= self.W(v, u).sum(-1)
            logit = (torch.bmm(u, v).squeeze(2)).sum(-1)
            #W is trainable; how to represent W ????? logit is the score
            #sum(-1): axis may be negative, in which case it counts from the last to the first axis.
            #ex. if an array has dimension 0, 1, 2; sum(-1) == sum(2); sum(-2)==sum(1)
            #print("logit: ", logit)
            #print("logit.sum(-1): ",logit.sum(-1) )
            #print("logit.shape: ", logit.shape)  # torch.Size([10])
            #input()
            #how to compute the score after calculating the attention??????
            logits.append(logit)
            #print("i= ", i) #i=0~4; evaluation i=0~4
        #print("logits before torch.stack: ",len(logits)) # len(logits)=5; the same as example net(?)
        logits = torch.stack(logits, 1) #Concatenates sequence of tensors along a new dimension. All tensors need to be of the same size.
        #print("logits after torch.stack: ", logits.shape) #torch.Size([10, 5]) ; the same as example net
        return logits
