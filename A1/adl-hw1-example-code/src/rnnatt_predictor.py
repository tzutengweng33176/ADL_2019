import torch
from base_predictor import BasePredictor
from modules import RNNAtt

#import matplotlib.pyplot as plt
#plt.switch_backend('agg')
#import matplotlib.ticker as ticker
#import numpy as np

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    #attentions.shape (opt_len, context_len)
    fig = plt.figure(figsize=(30, 30))
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions, cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels(input_sentence +
                       ['<EOS>'], rotation=90, )
    ax.set_yticklabels(output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.savefig('/home/tzutengweng/ADLHW/A1/adl-hw1-example-code/test.png')



class RNNAttPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, embedding, hidden_size=128,
                 dropout_rate=0.2, loss='BCELoss', margin=0, threshold=None,
                 similarity='inner_product', **kwargs):
        super(RNNAttPredictor, self).__init__(**kwargs)
        self.model = RNNAtt(embedding.size(1),hidden_size,dropout= dropout_rate, 
                                similarity=similarity)
        self.embedding = torch.nn.Embedding(embedding.size(0),
                                            embedding.size(1))
        self.embedding.weight = torch.nn.Parameter(embedding)
        #print("embedding.size(): ", embedding.size())
        # use cuda
        self.model = self.model.to(self.device)
        self.embedding = self.embedding.to(self.device)

        # make optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.learning_rate)
        #self.index2word= index2word
        self.loss = {
            'BCELoss': torch.nn.BCEWithLogitsLoss()
        }[loss]

    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embedding(batch['context'].to(self.device))
            options = self.embedding(batch['options'].to(self.device))
        logits, attn= self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        #if training==False:
            #context_sentence = [self.index2word[i] for i in batch['context'][0].numpy()]
            #candidate_sentence= [self.index2word[i] for i in batch['options'][0][0].numpy()]
            #showAttention(context_sentence, candidate_sentence, attn[0][0].cpu().numpy())
            #input()
        #print("logits.shape: ", logits.shape) #torch.Size([10, 5])
        #print("batch['labels']: ", batch['labels']) #torch.Size([10, 100]) in validation set, why??
        #-->because there are 100 candidate responses-->see make_dataset.py line 48,49,50  
        #training(bool) IS NOT used; when validating, training is false
        loss = self.loss(logits, batch['labels'].float().to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        #print("context", batch['context'].shape) #batch_size, context_len
        #print("options", batch['options'].shape) #batch_size, num_of_candidates=100, opt_len
        #print(batch['context'][0]) #index of words
        #print(batch['options'][0][0]) #(num_of_candidates, opt_len) #index of words
        
        #print(self.index2word[100])
        #context_sentence =[self.index2word[i] for i in batch['context'][0].numpy() if i!=83840] 
        #context_sentence =[self.index2word[i] for i in batch['context'][0].numpy()] 

        #candidate_sentence=[self.index2word[i] for i in batch['options'][0][0].numpy() if i!=83840]
        #candidate_sentence=[self.index2word[i] for i in batch['options'][0][0].numpy()]
        #print(context_sentence)
        #print(candidate_sentence)
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits, attn = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])

        #showAttention(context_sentence, candidate_sentence, attn[0][0].cpu().numpy())
        #plt.matshow(attn[0][0].cpu().numpy())
        #plt.savefig('/home/tzutengweng/ADLHW/A1/adl-hw1-example-code/test.png')
        #print("attn", attn[0][0].shape)# (batch, opt_len, context_len)
        #input()
        return logits
