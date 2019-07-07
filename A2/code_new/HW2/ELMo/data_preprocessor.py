from collections import Counter
import os
import codecs
import time
import argparse
import logging
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from modules.char_embedding import CharEmbedding
from modules.elmo import LSTMMP
import numpy as np
#you can also set oov to <UNK>       
def create_one_batch(x, word2id, char2id, config, oov='<OOV>', pad='<PAD>', sort=True):
  
  batch_size = len(x)
  lst = list(range(batch_size)) #a list of indices
  if sort:
    lst.sort(key=lambda l: -len(x[l])) #sort by decreasing lengths
  
  x = [x[i] for i in lst]
  #print(x) #a list of lists of tokens
  #print(x[0]) # a list of tokens
  #input()
  lens = [len(x[i]) for i in lst]
  #print(lens)
  #input()
  max_len = max(lens)

  if word2id is not None:
    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    #print(oov_id) #0
    #print(pad_id) #3
    #input()
    assert oov_id is not None and pad_id is not None
    batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    #print(batch_w.shape) #torch.Size([batch_size, max_sent_len])
    #input()
    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_w[i][j] = word2id.get(x_ij, oov_id) #turn this batch of training samples to word indices
  else:
    batch_w = None
  #print(batch_w)
  #input()
  if char2id is not None:
    bow_id, eow_id, oov_id, pad_id = char2id.get('<EOW>', None), char2id.get('<BOW>', None), char2id.get(oov, None), char2id.get(pad, None)

    assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None
    #max_characters_per_token is the word length!
    max_chars = 16 #16
    #print([w for i in lst for w in x[i]])# a list of tokens
    #print(len('only'))#4
    #input()
    #print(max([len(w) for i in lst for w in x[i]]) + 2)
    #assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
    #input() 
    batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)
    #the input to the char_embedding should be ``torch.tensor(shape=(batch_size, sentence_len, word_len), dtype=torch.int64)``
    for i, x_i in enumerate(x):
      #print(x_i) #a list of tokens
      for j, x_ij in enumerate(x_i):
        #print(x_ij) #a token
        #input()
        if x_ij == '<BOS>' or x_ij == '<EOS>': #if the token is <BOS> or <EOS>, see it as a character
          batch_c[i][j][0] = char2id.get(x_ij)
        elif x_ij not in word2id: #if this word is rare, map it to <OOV> or <UNK>
          batch_c[i][j][0] = oov_id
        elif len(x_ij)> max_chars: #if this word is longer than max_chars, map it to <OOV> or <UNK>
          batch_c[i][j][0]=oov_id  
        else:
          for k, c in enumerate(x_ij): 
            batch_c[i][j][k] = char2id.get(c, oov_id)
          
  else:
    batch_c = None
  #print(batch_c)
  #print(batch_c.shape) #torch.Size([32, 64, 50]) batch_size, max_sent_len, word_len(max_char)
  #input()
  masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]
  #masks!!
  for i, x_i in enumerate(x):
    for j in range(len(x_i)):
      masks[0][i][j] = 1
      if j + 1 < len(x_i):
        masks[1].append(i * max_len + j)
      if j > 0: 
        masks[2].append(i * max_len + j)

  assert len(masks[1]) <= batch_size * max_len
  assert len(masks[2]) <= batch_size * max_len

  masks[1] = torch.LongTensor(masks[1])
  masks[2] = torch.LongTensor(masks[2])

  return batch_w, batch_c, lens, masks



# shuffle training examples and create mini-batches
def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=True, sort=True, use_cuda=False):
  #len(x) is the number of training samples
  lst = perm or list(range(len(x)))
  #print(lst) #a list of indices of training samples
  #input()
  if shuffle:
    random.shuffle(lst)
  #print(lst) #shuffle the indices
  #input()
  if sort:
    lst.sort(key=lambda l: -len(x[l])) #sort by decreasing length of sentences
  
  x = [x[i] for i in lst]
  #print([l for l in x])
  #input()
  sum_len = 0.0
  batches_w, batches_c, batches_lens, batches_masks = [], [], [], []
  size = batch_size
  nbatch = (len(x) - 1) // size + 1 #nbatch is the number of batches
  for i in range(nbatch):
    start_id, end_id = i * size, (i + 1) * size #0~31, 32~63, ...
    bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config, sort=sort)
    #print(blens)#blens is a list of sentence lengths
    #print(bw.shape)#torch.Size([32, 64]) batch_size, max_sent_len
    #print(bc.shape) #torch.Size([32, 64, 50]) batch_size, max_sent_len, word_len
    #print(bmasks) #a list
    #print(len(bmasks)) # 3
    #input()
    sum_len += sum(blens)
    batches_w.append(bw) #append a batch of word
    batches_c.append(bc) #append a batch of chars
    batches_lens.append(blens)
    batches_masks.append(bmasks)

  if sort:
    #shuffle the batches  
    perm = list(range(nbatch))
    random.shuffle(perm)
    batches_w = [batches_w[i] for i in perm]
    batches_c = [batches_c[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batches_masks = [batches_masks[i] for i in perm]

  print("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
  return batches_w, batches_c, batches_lens, batches_masks


def get_truncated_char(dataset,min_count):
    char_count=Counter()
    for sample in dataset:
        for word in sample:
            #print(list(word))
            char_count.update(list(word))
            #print(char_count)
            #input()
    char_count=list(char_count.items())
    char_count.sort(key=lambda x: x[1], reverse=True) #sort the char by decreasing frequencies
    #print(char_count)
    #input()
    i=0
    for char, count in char_count:
        if count< min_count:
            break
        i+=1
    #print(char_count[i:])
    #print("Original character vocabulary size: ", len(char_count))
    #print("Truncated number of chars: ", sum([count for char, count in char_count[i:]]))
    #input()
    return char_count[:i]


def get_truncated_vocab(dataset, min_count):
#Python Counter!!!!
#Count word and chars! How????
  word_count=Counter()
  #print(word_count) #Counter()
  for sample in dataset:
      word_count.update(sample) #count the word in the sample and return a dict of word: count
      #print(list(word_count.items())) #a list of tuples ('word': count)
      #print(word_count) #Counter({'the': 4, 'was': 4, 'that': 3, 'is': 2, 'very': 2, 'good': 2, "n't": 2, 'it': 2, 'as': 2, 'original': 2, '.': 2, 'us': 2, 'and': 2, '<BOS>': 1, 'This': 1, 'a': 1, 'restaurant': 1, ',': 1, 'but': 1, 'I': 1, 'do': 1, 'think': 1, 'Our': 1, 'first': 1, 'visit': 1, 'manager': 1, 'rude': 1, 'to': 1, 'apparent': 1, 'this': 1, 'The': 1, 'nephew': 1, 'of': 1, 'owner': 1, 'bussing': 1, 'tables': 1, 'saw': 1, 'we': 1, 'were': 1, 'upset': 1, 'put': 1, 'in': 1, 'touch': 1, 'with': 1, 'his': 1, 'uncle': 1})
      #input()
  word_count= list(word_count.items())
  word_count.sort(key=lambda x: x[1], reverse=True) #sort the word by decreasing frequencies
  #print(word_count)
  #input()
  i=0
  for word, count in word_count:
      if count< min_count:
          break
      i+=1
  #print(word_count[i:])
  print("Original vocabulary size: ", len(word_count))
  print("Truncated number of words: ", sum([count for word, count in word_count[i:]]))
  #input()
  return word_count[:i]

def break_sentence(sentence, max_sent_len):
  ret= []
  cur=0
  length= len(sentence)
  #print(length)
  while cur< length:
    if cur + max_sent_len+5 >= length: #we keep 5 to have the least sentence length=5
      ret.append(sentence[cur:length])
      break
    ret.append(sentence[cur:min(cur+max_sent_len, length)])
    cur +=max_sent_len #the pointer move 64 once
    #print(cur)
  
  return ret    

def read_corpus(path, max_sent_len=64):
  training_samples=[]
  with open(path, 'r') as fin:
      for line in fin:
          sent=["<BOS>"]
          sent+=(line.split())
          sent.append("<EOS>")
          dataset= break_sentence(sent, max_sent_len)
          training_samples += dataset
          #print(training_samples)
          #input()
          #print(line.split()) #['Quick', '&', 'courteous', '!', 'Got', 'a', 'replacement', 'key', 'for', 'my', 'car', 'for', 'less', 'than', 'dealer', '.', 'Open', 'on', 'the', 'weekend', ':)']
         #innut()
  #print(len(training_samples))
  return training_samples

class Model(nn.Module):
  def __init__(self, config,num_of_chars, num_of_words, use_cuda=False):
    super(Model, self).__init__() 
    self.use_cuda = use_cuda
    self.config = config

    self.token_embedder = CharEmbedding(num_embeddings=num_of_chars, embedding_dim=16, padding_idx=4, 
            conv_filters=[(1, 32), (2, 64), (3, 128), (4, 128), (5, 256), (6, 256), (7, 512)], n_highways=2, projection_size=512)

    self.encoder = LSTMMP(input_size= 512 ,num_layers=2 ,hidden_size=2048, projection_size=512 )

    #self.output_dim = config['encoder']['projection_dim']
   #https://pytorch.org/docs/stable/nn.html#adaptivelogsoftmaxwithloss
   #torch.nn.AdaptiveLogSoftmaxWithLoss
    self.classify_layer = nn.AdaptiveLogSoftmaxWithLoss(512, n_classes= num_of_words, cutoffs= [100, 1000, 10000])

  def forward(self, w, c, mask_package):
    """
    w : batch_size, max_sent_len
    c: batch_size, max_sent_len, word_len
    mask: a list

    """
   # classifier_name = self.config['classifier']['name'].lower()
    if self.use_cuda:
      c = c.cuda()
    #if self.training and classifier_name == 'cnn_softmax' or classifier_name == 'sampled_softmax':
    #  self.classify_layer.update_negative_samples(word_inp, chars_inp, mask_package[0])
    #  self.classify_layer.update_embedding_matrix()

    token_embedding = self.token_embedder(c)
    #dropout here!!
    #print(token_embedding)
    #print(token_embedding.shape) # batch, max_sent_len, 512 -->512 will be the input size of LSTM
    #input()
    token_embedding = F.dropout(token_embedding, 0.1, self.training)

    #encoder_name = self.config['encoder']['name'].lower()
    forward, backward = self.encoder(token_embedding)
    forward= forward[1]
    backward=backward[1]
    #forward and backward have shape batch, seq_len-1, projecion_size= 512
    #print(forward.shape) #batch_size, seq_len-1, projection_size
    forward= F.dropout(forward, 0.1, self.training)
    backward= F.dropout(backward, 0.1, self.training)
    #input()
    #The Variable API has been deprecated: Variables are no longer necessary to use autograd with tensors. Autograd automatically supports Tensors with requires_grad set to True.
    if self.use_cuda:
      w = w.cuda()
      forward = forward.cuda()
      backward = backward.cuda()
    
    #print(w.shape) #batch_size, seq_len
    b_size, seq_len= w.shape
    
    mask1= torch.arange(seq_len-1).long() # 0~seq_len-2
    mask2 = torch.arange(1, seq_len).long() #1 ~seq_len-1
    
    mask1=mask1.cuda() if self.use_cuda else mask1
    mask2= mask2.cuda() if self.use_cuda else mask2
    #print(forward.contiguous().shape) #batch, seq_len-1, projection_size=512
    #print(forward.contiguous().view(-1, 512).shape) # batch * (seq_len-1), 512
    #print(w.contiguous().view(-1).shape) #batch*seq_len
    #print(mask1) #0 ~62
    #print(mask2) #1 ~63
    #print(mask1.shape) 
    #print(mask2.shape)
    #print(w.index_select(1, mask2).shape)  #select Hello, World,!, <EOS>
    #print(w.index_select(1, mask1).shape) # select <BOS>, Hello, World, !
    #print(w.index_select(1, mask1).contiguous().view(-1))
    #input()
    #input sentence= "<BOS>, Hello, World,!"
    #forward_x = forward.contiguous().view(-1, 512).index_select(0, mask1) #predicted sentence
    forward_x = forward.contiguous().view(-1, 512) #predicted sentence
    forward_y = w.index_select(1, mask2).contiguous().view(-1)#the target sentence "Hello, World,! , <EOS>"
    #print(forward_x)
    #print(forward_x.shape) # batch*(seq_len -1 ) ,projection_size= 512
    #print(forward_y) #target 
    #print(forward_y.shape) # batch *(seq_len -1)!!!!
    #input()
    #backward_x = backward.contiguous().view(-1, 512).index_select(0, mask2)
    backward_x = backward.contiguous().view(-1, 512)
    backward_y = w.index_select(1, mask1).contiguous().view(-1)
    #about index_select...
    #https://pytorch.org/docs/stable/torch.html#torch.index_select 
    #the classifier layer is shared by forward and backward output
    return self.classify_layer(forward_x, forward_y), self.classify_layer(backward_x, backward_y)

  def save_model(self, path, save_classify_layer):
    torch.save(self.token_embedder.state_dict(), os.path.join(path, 'token_embedder.pkl'))    
    torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pkl'))
    if save_classify_layer:
      torch.save(self.classify_layer.state_dict(), os.path.join(path, 'classifier.pkl'))

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl')))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl')))
    self.classify_layer.load_state_dict(torch.load(os.path.join(path, 'classifier.pkl')))

def eval_model(model, valid):
  model.eval()
  
  total_loss, total_tag = 0.0, 0
  valid_w, valid_c, valid_lens, valid_masks = valid
  for w, c, lens, masks in zip(valid_w, valid_c, valid_lens, valid_masks):
    #print(w.shape) #batch_size, seq_len
    #print(c.shape) #batch_size, seq_len, word_len
    #input()
    loss_forward, loss_backward = model.forward(w, c, masks)
    loss_forward= loss_forward[1]
    loss_backward= loss_backward[1]
    #total_loss += loss_forward.data[0]
    total_loss += float(loss_forward.item())
    n_tags = sum(lens)
    total_tag += n_tags
  model.train()
  return np.exp(total_loss / total_tag)


def train_model(epoch, args, model, optimizer,
                train, valid,  best_train, best_valid, test_result):
  """
  Training model for one epoch

  :return:
  """
  model.train()

  total_loss, total_tag = 0.0, 0
  cnt = 0
  start_time = time.time()

  train_w, train_c, train_lens, train_masks = train

  lst = list(range(len(train_w)))
  random.shuffle(lst)
  
  train_w = [train_w[l] for l in lst]
  train_c = [train_c[l] for l in lst]
  train_lens = [train_lens[l] for l in lst]
  train_masks = [train_masks[l] for l in lst]

  for w, c, lens, masks in zip(train_w, train_c, train_lens, train_masks):
    cnt += 1
    model.zero_grad() #the same as optimizer.zero_grad()
    #print(w.shape)
    #print(c.shape)
    #print(len(masks)) #3
    #input()
    loss_forward, loss_backward = model.forward(w, c, masks)
    #print(loss_forward[1]) #tensor(8.5735, device='cuda:0', grad_fn=<MeanBackward1>)
    #print(loss_backward[1]) #tensor(8.5533, device='cuda:0', grad_fn=<MeanBackward1>)
    loss_forward= loss_forward[1]
    loss_backward= loss_backward[1]
    #input()
    #do we need to accumulate gradients if CUDA run out of memory? 
    #https://pytorch.org/docs/stable/notes/faq.html
    loss = (loss_forward + loss_backward) / 2.0
    if cnt*32 %1024 ==0:
      print("NLL Loss: ", loss) #negative log likelihood loss
    #total_loss += loss_forward.data[0] #ORIGINAL
    #print(loss_forward.item())
    #print(float(loss_forward.item()))
    #input()
    total_loss += float(loss_forward.item())
    n_tags = sum(lens)
    total_tag += n_tags
    #https://discuss.pytorch.org/t/why-do-we-need-to-set-the-gradients-manually-to-zero-in-pytorch/4903/20
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
    optimizer.step()
    if cnt * 32 % 1024 == 0:
      print("Epoch={} iter={} lr={:.6f} train_ppl={:.6f} time={:.2f}s".format(
        epoch, cnt, optimizer.param_groups[0]['lr'],
        np.exp(total_loss / total_tag), time.time() - start_time
      ))
      start_time = time.time()

    if cnt % 1024 == 0 or cnt % len(train_w) == 0:
      if valid is None:
        train_ppl = np.exp(total_loss / total_tag)
        print("Epoch={} iter={} lr={:.6f} train_ppl={:.6f}".format(
          epoch, cnt, optimizer.param_groups[0]['lr'], train_ppl))
        if train_ppl < best_train:
          best_train = train_ppl
          print("New record achieved on training dataset!")
          model.save_model(args.model, False)      
      else:
        valid_ppl = eval_model(model, valid)
        print("Epoch={} iter={} lr={:.6f} valid_ppl={:.6f}".format(
          epoch, cnt, optimizer.param_groups[0]['lr'], valid_ppl))

        if valid_ppl < best_valid:
          model.save_model(args.model, False)
          best_valid = valid_ppl
          print("New record of ppl achieved on validation set!")

          #if test is not None:
          #  test_result = eval_model(model, test)
          #  logging.info("Epoch={} iter={} lr={:.6f} test_ppl={:.6f}".format(
          #    epoch, cnt, optimizer.param_groups[0]['lr'], test_result))
  return best_train, best_valid, test_result


def train():
    parser= argparse.ArgumentParser(description='Data preprocessing and Self-made ELMo training')
    parser.add_argument('--train_path', required=True, help='The path to the training file.')
    parser.add_argument('--valid_path', help='The path to the development file.')
    parser.add_argument('--model', required=True, help="path to save model")

    parser.add_argument("--batch_size", "--batch", type=int, default=32, help='the batch size.')
    parser.add_argument("--max_epoch", type=int, default=10, help='the maximum number of iteration.')
    
    args=parser.parse_args()
    #print(args)
    #print(args.train_path)
    #print(args.model)
    #input()
    train_data=read_corpus(args.train_path)
    valid_data=read_corpus(args.valid_path) 
    #train_data=read_corpus("/home/tzutengweng/ADLHW/A2/code_new/HW2/data/language_model/1M_corpus_tokenized.txt")
    #valid_data=read_corpus("/home/tzutengweng/ADLHW/A2/code_new/HW2/data/language_model/valid_5k_corpus_tokenized.txt") 
    #print(train_data) # a list of split sentences
    vocab= get_truncated_vocab(train_data, 3) #min-count=3
    char = get_truncated_char(train_data, 1000) 
    #vocab is a list of tuples (word, count)
    #how to map word to <UNK> if its count is less than 3? 
    #we need a word_lexicon, its a dict of word to index
    word_lexicon={}
    #print(word_lexicon) #{'<OOV>': 0, '<BOS>': 1, '<EOS>': 2, '<PAD>': 3}
    #input()
    for word, count in vocab:
        if word not in word_lexicon:
            word_lexicon[word]= len(word_lexicon)
    #print(word_lexicon)
    
    for special_word in ['<OOV>', '<PAD>', '<UNK>']:
        if special_word not in word_lexicon:
            word_lexicon[special_word]= len(word_lexicon)
    
    print("Size of word lexicon: ", len(word_lexicon))
    #input()
    #Character lexicon, a dict of character to index
    char_lexicon={}
    for special_char in ['<BOS>', '<EOS>', '<OOV>', '<UNK>', '<PAD>', '<BOW>', '<EOW>']:
        if special_char not in char_lexicon:
            char_lexicon[special_char]= len(char_lexicon)
    
    for char, count in char:
        if char not in char_lexicon:
            char_lexicon[char]=len(char_lexicon)
    print("Size of char lexicon: ", len(char_lexicon))
    num_of_chars= len(char_lexicon)
    num_of_words= len(word_lexicon)
                    #input()
    #print(char_lexicon)
    #input()

    try:
      os.makedirs(args.model)
    except OSError as exception:
      if exception.errno != errno.EEXIST:
        raise
    #save word.dic and char.dic
    #input()
    with codecs.open(os.path.join(args.model, 'word.dic'), 'w', encoding='utf-8') as fpo:
        for w, i in word_lexicon.items():
            print('{0}\t{1}'.format(w, i), file=fpo)

    with codecs.open(os.path.join(args.model, 'char.dic'), 'w', encoding='utf-8') as fpo:
        for c, i in char_lexicon.items():
            print('{0}\t{1}'.format(c, i), file=fpo)
    
    config=None
    batch_size=args.batch_size
    use_cuda= True
    train= create_batches(train_data, batch_size, word_lexicon, char_lexicon, config, use_cuda=True)
    
    if valid_data is not None:
      valid = create_batches(
        valid_data, batch_size, word_lexicon, char_lexicon, config, sort=False, shuffle=False, use_cuda=use_cuda)
    else:
      valid=None
    #print(train)
    #input()
    need_grad = lambda x: x.requires_grad
    model = Model(config,num_of_chars ,num_of_words, use_cuda)
    if use_cuda:
      model = model.cuda()
    
    optimizer = optim.Adam(filter(need_grad, model.parameters()), lr=0.001)
    
    best_train = 1e+8
    best_valid = 1e+8
    test_result = 1e+8
    
    for epoch in range(args.max_epoch):
      best_train, best_valid, test_result= train_model(epoch,args ,model, optimizer,
                                                      train, valid,  best_train, best_valid, test_result)

if __name__ == "__main__":
    train()
