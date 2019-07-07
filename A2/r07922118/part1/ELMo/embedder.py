import numpy as np
from .sentstoELMo import TestEmbedder 
import os
class Embedder:
    """
    The class responsible for loading a pre-trained ELMo model and provide the ``embed``
    functionality for downstream BCN model.

    You can modify this class however you want, but do not alter the class name and the
    signature of the ``embed`` function. Also, ``__init__`` function should always have
    the ``ctx_emb_dim`` parameter.
    """

    def __init__(self, n_ctx_embs, ctx_emb_dim):
        """
        The value of the parameters should also be specified in the BCN model config.
        """
        self.n_ctx_embs = n_ctx_embs
        self.ctx_emb_dim = ctx_emb_dim
        # TODO

    def __call__(self, sentences, max_sent_len):
        """
        Generate the contextualized embedding of tokens in ``sentences``.

        Parameters
        ----------
        sentences : ``List[List[str]]``
            A batch of tokenized sentences.
        max_sent_len : ``int``
            All sentences must be truncated to this length.

        Returns
        -------
        ``np.ndarray``
            The contextualized embedding of the sentence tokens.

            The ndarray shape must be
            ``(len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim)``
            and dtype must be ``np.float32``.
        """
        #print(sentences) # a list of sentences split to tokens
        #we should padd the sentence to the same length
        len_of_sent= min(max(map(len, sentences)), max_sent_len)
        #print(len(sentences)) #batch_size
        #print(len_of_sent) #33
        processed_sentences=[]
        #cur=0
        for sent in sentences:
            #while (len(sent)< len_of_sent):
                #pad the sentences with <pad>
            #    sent.append('<PAD>')
            if (len(sent)>len_of_sent):#truncate the sentence to the len_of_sent
                sent= sent[:len_of_sent]

            processed_sentences.append(sent)
        #print(list(map(len, processed_sentences))) #success
        print("Load trained ELMo model from :", os.path.join(os.getcwd(), "ELMo/test_1M_1_epoch_0422"))
        #input()
        e= TestEmbedder(os.path.join(os.getcwd(), "ELMo/test_1M_1_epoch_0422"), len(sentences))
        a=e.sents2elmo(processed_sentences, -2)
        #we should padd the sentence to the same length
        #print(a)
        #print(len(a)) #32, batch_size
        #print(map(len, sentences)) #<map object at 0x7f4278135358>
        #print(max(map(len, sentences))) #33 ? WHY???
        #print(list(map(len, sentences))) #[19, 19, 9, 33, 19, 3, 7, 13, 18, 16, 5, 8, 11, 16, 29, 23, 9, 21, 22, 7, 15, 20, 31, 22, 21, 21, 10, 16, 22, 10, 20, 6]
        #print(a.shape) #sents2elmo will return a list
        #print(a[1])# a numpy array
        #print(a[0].shape) #  (3, 33, 1024)
        #input()
        b= np.stack(a, axis=0) #ValueError('all input arrays must have the same shape',)
        #print(b[2])#
        #print(b[3])
        #print(b.shape)#(32,3 , 33, 1024) ``(len(sentences), min(max(len, sentences), max_sent_len), self.ctx_emb_dim)``
        #print(b.dtype) #float32
        b=b.transpose(0, 2, 1, 3)
        #print(b.shape) #32, 33, 3, 1024
        #input()
        # TODO
        return b
        #return np.zeros(
        #    (len(sentences), min(max(map(len, sentences)), max_sent_len), self.n_ctx_embs, self.ctx_emb_dim), dtype=np.float32)
