import torch
from base_predictor import BasePredictor
from modules import BestRNNAtt


class BestRNNAttPredictor(BasePredictor):
    """

    Args:
        dim_embed (int): Number of dimensions of word embedding.
        dim_hidden (int): Number of dimensions of intermediate
            information embedding.
    """

    def __init__(self, embedding, hidden_size=128,
                 dropout_rate=0.2, loss='BCELoss', margin=0, threshold=None,
                 similarity='inner_product', **kwargs):
        super(BestRNNAttPredictor, self).__init__(**kwargs)
        self.model = BestRNNAtt(embedding.size(1),hidden_size,dropout= dropout_rate, 
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

        self.loss = {
            'BCELoss': torch.nn.BCEWithLogitsLoss()
        }[loss]

    def _run_iter(self, batch, training):
        with torch.no_grad():
            context = self.embedding(batch['context'].to(self.device))
            options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        #print("logits.shape: ", logits.shape) #torch.Size([10, 5])
        #print("batch['labels']: ", batch['labels']) #torch.Size([10, 100]) in validation set, why??
        #-->because there are 100 candidate responses-->see make_dataset.py line 48,49,50  
        #training(bool) IS NOT used; when validating, training is false
        loss = self.loss(logits, batch['labels'].float().to(self.device))
        return logits, loss

    def _predict_batch(self, batch):
        context = self.embedding(batch['context'].to(self.device))
        options = self.embedding(batch['options'].to(self.device))
        logits = self.model.forward(
            context.to(self.device),
            batch['context_lens'],
            options.to(self.device),
            batch['option_lens'])
        return logits
