# How to run

```
1.First, download the pre-trained model and preprocessed word-embedding

bash ./download.sh 

2.After downloading, predict using rnn, rnnatt, best by

bash ./rnn.sh /path/to/test.json /path/to/predict_rnn.csv

bash ./attention.sh /path/to/test.json /path/to/predict_rnnatt.csv

bash ./best.sh /path/to/test.json /path/to/predict_best.csv


```
#How to train my model

```
I simply concatenate all the utterances together. 

RNN without attention: 
I set hidden size as 128 and use nn.Bilinear to calculate the score of each option.  
I used max-pooling to deal with the output of RNN for both context and each option. 


RNN with attention: 
I still set hidden size to 128 and use max-pooling to deal with the output of first RNN layer for both context and each option. Then I calculate the attention energies by torch.bmm(opt_output, context_outputs.transpose(1,2)), which means that I use a word in the context to calculate attention energy with every word in each option and will form a attention energy matrix of shape(batch, opt_len, context_len). Then I applied a softmax along the context dimension to get the attention weights of context.  
Apply the attention weight to the context and then concatenate the resulting vector with 1st RNN output of each option. Then I put the concatenated big vector into 2nd layer of RNN to get the output_2.

I calculate the score of each option by dot product of output of the context and output_2(both outputs were dealt with max-pooling after the RNN.).

```
#How to plot the figures 
```
1. I use the embedding.word_dict to get the word-to-index dictionary and then I turn that to index-to-word dictionary by:

    index2word =  {v: k for k, v in word2index.items()}

2.I pass the index2word to the Predictor class as a lookup table. 

3. Then I modify the _run_iter function in the rnnatt_predictor.py to the below....part of the code was modified from PyTorch tutorial website.

if training==False:
     context_sentence = [self.index2word[i] for i in batch['context'][0].numpy()]
     candidate_sentence= [self.index2word[i] for i in batch['options'][0][0].numpy()]
     showAttention(context_sentence, candidate_sentence, attn[0][0].cpu().numpy())

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

Reference: https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html


```
