import torch
from base_predictor import BasePredictor
from modules import RNNAtt


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
        loss = self.loss(logits, batch['labels'].float().to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits, attn = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        return logits
