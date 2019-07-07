#!/usr/bin/env python
import os
import random
import torch
import torch.nn as nn
import logging
from .modules.elmo import LSTMMP
from .modules.char_embedding import CharEmbedding


def create_one_batch(x, word2id, char2id, config, oov='<OOV>', pad='<PAD>', sort=True):
  """
  Create one batch of input.

  :param x: List[List[str]]
  :param word2id: Dict | None
  :param char2id: Dict | None
  :param config: Dict
  :param oov: str, the form of OOV token.
  :param pad: str, the form of padding token.
  :param sort: bool, specify whether sorting the sentences by their lengths.
  :return:
  """
  batch_size = len(x)
  # lst represents the order of sentences
  lst = list(range(batch_size))
  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  # shuffle the sentences by
  x = [x[i] for i in lst]
  lens = [len(x[i]) for i in lst]
  max_len = max(lens)

  # get a batch of word id whose size is (batch x max_len)
  if word2id is not None:
    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    assert oov_id is not None and pad_id is not None
    batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_w[i][j] = word2id.get(x_ij, oov_id)
  else:
    batch_w = None

  # get a batch of character id whose size is (batch x max_len x max_chars)
  if char2id is not None:
    bow_id, eow_id, oov_id, pad_id = [char2id.get(key, None) for key in ('<EOW>', '<BOW>', oov, pad)]

    assert bow_id is not None and eow_id is not None and oov_id is not None and pad_id is not None

    max_chars = 16
    #assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars

    batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)

    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        if x_ij == '<BOS>' or x_ij == '<EOS>':
          batch_c[i][j][0] = char2id.get(x_ij)
        elif x_ij not in word2id:
          batch_c[i][j][0]= oov_id
        elif len(x_ij)>max_chars:
          batch_c[i][j][0]= oov_id
        else:
          for k, c in enumerate(x_ij):
            batch_c[i][j][k] = char2id.get(c, oov_id)
  else:
    batch_c = None

  # mask[0] is the matrix (batch x max_len) indicating whether
  # there is an id is valid (not a padding) in this batch.
  # mask[1] stores the flattened ids indicating whether there is a valid
  # previous token
  # mask[2] stores the flattened ids indicating whether there is a valid
  # next token
  masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

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
def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=True, sort=True, text=None):
  """

  :param x: List[List[str]]
  :param batch_size:
  :param word2id:
  :param char2id:
  :param config:
  :param perm:
  :param shuffle:
  :param sort:
  :param text:
  :return:
  """
  lst = perm or list(range(len(x)))
  if shuffle:
    random.shuffle(lst)

  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  if text is not None:
    text = [text[i] for i in lst]

  sum_len = 0.0
  batches_w, batches_c, batches_lens, batches_masks, batches_text = [], [], [], [], []
  size = batch_size
  nbatch = (len(x) - 1) // size + 1
  for i in range(nbatch):
    start_id, end_id = i * size, (i + 1) * size
    bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config, sort=sort)
    sum_len += sum(blens)
    batches_w.append(bw)
    batches_c.append(bc)
    batches_lens.append(blens)
    batches_masks.append(bmasks)
    if text is not None:
      batches_text.append(text[start_id: end_id])

  if sort:
    perm = list(range(nbatch))
    random.shuffle(perm)
    batches_w = [batches_w[i] for i in perm]
    batches_c = [batches_c[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batches_masks = [batches_masks[i] for i in perm]
    if text is not None:
      batches_text = [batches_text[i] for i in perm]

  logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
  if text is not None:
    return batches_w, batches_c, batches_lens, batches_masks, batches_text
  return batches_w, batches_c, batches_lens, batches_masks


class Model(nn.Module):
  def __init__(self, config, num_of_chars, num_of_words, use_cuda=False):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.config = config

    self.token_embedder = CharEmbedding( num_embeddings=num_of_chars, embedding_dim=16, padding_idx=4,
            conv_filters=[(1, 32), (2, 64), (3, 128), (4, 128), (5, 256), (6, 256), (7, 512)], n_highways=2, projection_size=512)
    self.encoder = LSTMMP(input_size=512, num_layers=2, hidden_size=2048, projection_size=512)


  def forward(self, w, c, mask_package):
      #print("c.shape", c.shape) #batch_size, seq_len, word_len
      #print("w.shape", w.shape) #batch_size, seq_len
      #print("c", c) #idx of chars
      #print("w", w) #idx of words
      #input()
      if self.use_cuda:
          c=c.cuda()
      token_embedding = self.token_embedder(c)
      #print(token_embedding.shape) #batch_size, seq_len, projection_size
      #you also have to reverse the token_embedding here
      #we have included <BOS> and <EOS>, so we need to remove them here
      #use mask and index_select here!!!
      f, b = self.encoder(token_embedding)
      #f[0] the 1st layer, f[1] the 2nd layer
      #print(f.shape) #2, batch_size, seq_len-1, 512
      #print(b.shape) #the same as above
      #for f, we remove the first token
      #for b we remove the last token
      sz= f.size()  #2, batch_size, seq_len-1, projection_size
      #print("sz[2]: ", sz[2])
      token_idx = torch.arange(1, sz[2]).long()
      back_idx= torch.arange(1, sz[2]).long()
      for_idx= torch.arange(0, sz[2]-1).long()
      
      token_idx= token_idx.cuda()
      for_idx= for_idx.cuda()
      back_idx= back_idx.cuda()
      #print(token_idx)
      #print(for_idx)
      #print(back_idx)
      #print(sz)
      #input()
      token_embedding= token_embedding.index_select(1, token_idx)
      f= f.index_select(2, for_idx)
      b=b.index_select(2, back_idx)
      #print(token_embedding.shape) #batch_size, seq_len-2 (remove <BOS> and <EOS>), 512
      #print(f.shape) #2, batch_size, seq_len-2, 512
      #print(b.shape)
      #sz_1 = f.size()
      encoder_output = torch.cat([f, b], dim=3)
      #print(encoder_output.shape) # 2, batch_size, seq_len-2, 1024
      #input()
      #token_embedding will be the 1st layer
      token_embedding = torch.cat(
        (token_embedding, token_embedding), dim=2).unsqueeze(0)
      #print(token_embedding.shape) # 1, batch_size, seq_len-2, 1024
      #LSTMMP will be the 2nd and 3rd layer
      encoder_output = torch.cat(
         (token_embedding, encoder_output), dim=0)
      #print(encoder_output.shape) # 3, batch_size, seq_len-2, 1024
      #input()

      return encoder_output

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl'),
                                                   map_location=lambda storage, loc: storage))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl'),
                                            map_location=lambda storage, loc: storage))
