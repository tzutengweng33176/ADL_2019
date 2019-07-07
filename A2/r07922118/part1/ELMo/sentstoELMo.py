#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import codecs
import random
import logging
import json
import torch
from .frontend import create_one_batch
from .frontend import Model
import numpy as np

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)-15s %(levelname)s: %(message)s')


def read_list(sents, max_chars=None):
    """
    read raw text file. The format of the input is like, one sentence per line
    words are separated by '\t'

    :param path:
    :param max_chars: int, the number of maximum characters in a word, this
      parameter is used when the model is configured with CNN word encoder.
    :return:
    """
    dataset = []
    textset = []
    for sent in sents:
        data = ['<BOS>']
        text = []
        for token in sent:
            text.append(token)
            #if max_chars is not None and len(token) + 2 > max_chars:
            #    token = token[:max_chars - 2]
            data.append(token)
        #data.append('<PAD>')
        data.append('<EOS>')
        dataset.append(data)
        textset.append(text)
    return dataset, textset


def recover(li, ind):
    # li[piv], ind = torch.sort(li[piv], dim=0, descending=(not unsort))
    dummy = list(range(len(ind)))
    #print(ind) #[0, 1, ...,  31]
    #print(len(ind)) #batch_size
    #print(dummy) 
    dummy.sort(key=lambda l: ind[l])
    #print(dummy) #[0, 1, ..., 31]
    li = [li[i] for i in dummy]
    #print(li)
    #input()
    return li


# shuffle training examples and create mini-batches
def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=False, sort=True, text=None):
    ind = list(range(len(x)))
    #print(ind) # [0, 1, 2, ..., batch_size -1]
    lst = perm or list(range(len(x)))
    #print(lst) #[0, 1, 2, ..., batch_size-1 ]
    if shuffle:
        random.shuffle(lst)

    if sort:
        lst.sort(key=lambda l: -len(x[l]))

    x = [x[i] for i in lst]
    ind = [ind[i] for i in lst]
    if text is not None:
        text = [text[i] for i in lst]
    #print(text) #a list of lists of tokens
    sum_len = 0.0
    batches_w, batches_c, batches_lens, batches_masks, batches_text, batches_ind = [], [], [], [], [], []
    size = batch_size
    nbatch = (len(x) - 1) // size + 1
    #print("nbatch: ", nbatch) #1
    #input()
    for i in range(nbatch):
        start_id, end_id = i * size, (i + 1) * size
        bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config, sort=sort)
        sum_len += sum(blens)
        batches_w.append(bw)
        batches_c.append(bc)
        batches_lens.append(blens)
        batches_masks.append(bmasks)
        batches_ind.append(ind[start_id: end_id]) #different from create_batches in data_preprocessor
        if text is not None:       # you will execute this line!
            batches_text.append(text[start_id: end_id])

    if sort:
        perm = list(range(nbatch))
        random.shuffle(perm)
        batches_w = [batches_w[i] for i in perm]
        batches_c = [batches_c[i] for i in perm]
        batches_lens = [batches_lens[i] for i in perm]
        batches_masks = [batches_masks[i] for i in perm]
        batches_ind = [batches_ind[i] for i in perm]
        if text is not None:
            batches_text = [batches_text[i] for i in perm]

    print("{} batches, avg len: {:.1f}".format(
        nbatch, sum_len / len(x)))
    recover_ind = [item for sublist in batches_ind for item in sublist]
    #what is recover_ind???
    #print(recover_ind) # [0, 1, ..., batch_size-1]
    #input()
    if text is not None:
        return batches_w, batches_c, batches_lens, batches_masks, batches_text, recover_ind
    return batches_w, batches_c, batches_lens, batches_masks, recover_ind


class TestEmbedder(object):
    def __init__(self, model_dir, batch_size=32):
        self.model_dir = model_dir
        self.model, self.config = self.get_model()
        self.batch_size = batch_size

    def get_model(self):
        # torch.cuda.set_device(1)
        self.use_cuda = torch.cuda.is_available()
        # load the model configurations
        #print(self.model_dir)
        #args2 = dict2namedtuple(json.load(codecs.open(
        #    os.path.join(self.model_dir, 'config.json'), 'r', encoding='utf-8')))

        #with open(os.path.join(self.model_dir, args2.config_path), 'r') as fin:
        #    config = json.load(fin)
        config=None
        # For the model trained with character-based word encoder.
        self.char_lexicon = {}
        with codecs.open(os.path.join(self.model_dir, 'char.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')
                token, i = tokens
                self.char_lexicon[token] = int(i)
        num_of_chars = len(self.char_lexicon)
        print('char embedding size: ' +
                        str(num_of_chars))

        # For the model trained with word form word encoder.
        self.word_lexicon = {}
        with codecs.open(os.path.join(self.model_dir, 'word.dic'), 'r', encoding='utf-8') as fpi:
            for line in fpi:
                tokens = line.strip().split('\t')
                if len(tokens) == 1:
                    tokens.insert(0, '\u3000')  #'\u3000'?? what is this??
                token, i = tokens
                self.word_lexicon[token] = int(i)
        num_of_words = len(self.word_lexicon)
        print('word embedding size: ' +
                        str(num_of_words))

        
        # instantiate the model
        model = Model(config, num_of_chars, num_of_words, self.use_cuda)

        if self.use_cuda:
            model.cuda()

        logging.info(str(model))
        model.load_model(self.model_dir)

        # read test data according to input format

        # configure the model to evaluation mode.
        model.eval()
        return model, config

    def sents2elmo(self, sents, output_layer=-1):
        read_function = read_list

        test, text = read_function(sents, 16) #16

        #print("test: ", test) #[['<bos>', 'I', 'am ', 'Tom', '<eos>'], ['<bos>', 'Hello', 'Tom', 'how', 'are', 'you', '?', 'I', 'am', 'fine', '<eos>']] OK
        #print("text: ", text)  # [['I', 'am ', 'Tom'], ['Hello', 'Tom', 'how', 'are', 'you', '?', 'I', 'am', 'fine']] OK
        #print("self.word_lexicon: ", self.word_lexicon) #a word2index dictionary
        #print("self.char_lexicon: ", self.char_lexicon) #a char2index dictionary
        #input()
        #print("self.config: ", self.config)# the same as in cnn_50_100_512_4096_sample.json
        #print("self.batch_size: ", self.batch_size) # 64
        # create test batches from the input data.
        test_w, test_c, test_lens, test_masks, test_text, recover_ind = create_batches(
            test, self.batch_size, self.word_lexicon, self.char_lexicon, self.config, text=text)
        #why do we need to create batches????
        cnt = 0

        after_elmo = []
        for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
            #print(w.shape) #batch_size, seq_len 
            #print(c.shape) #batch_size, seq_len, word_len
            output = self.model.forward(w, c, masks)
            #print("output: ", output.shape) #torch.Size([3, batch_size, seq_len-2, 1024])
            #print("texts: ", texts) #len=batch_size [['It', "'s", 'hard', 'to', 'fairly', 'judge', 'a', 'film', 'like', 'RINGU', 'when', 'you', "'", 've', 'seen', 'the', 'remake', 'first', '.', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>', '<pad>'], .... ] 
            #print("len(texts): ", len(texts))  #batch_size
            #input()
            for i, text in enumerate(texts):
                #print("len(text): ", len(text))
            #remove <BOS> and <EOS>-->in my model, I have already removed <BOS> and <EOS>
            #so don't have to use [:, i, 1:lens[i]-1, :].data here~~
                #print("len[i]", lens[i]) #len(text) +2
                data = output[:, i,:, :].data
                #print(data.shape) # 3, seq_len-2, 1024
                #input()
                if self.use_cuda:
                    data = data.cpu()
                data = data.numpy()

                if output_layer == -1: #average of three layers!
                    payload = np.average(data, axis=0)
                elif output_layer == -2:
                    payload = data
                else:
                    payload = data[output_layer]
                after_elmo.append(payload)

                cnt += 1
                if cnt % 1000 == 0:
                    print('Finished {0} sentences.'.format(cnt))
        #print("after_elmo: ", after_elmo) #still numpy array
        after_elmo = recover(after_elmo, recover_ind)
        return after_elmo
