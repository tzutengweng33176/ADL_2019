import torch
import torch.utils.data.dataloader
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm


class BasePredictor():
    def __init__(self,
                 batch_size=10,
                 max_epochs=10,
                 valid=None,
                 device=None,
                 metrics={},
                 learning_rate=1e-3,
                 max_iters_in_epoch=1e20,
                 grad_accumulate_steps=1):
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.valid = valid
        self.metrics = metrics
        self.learning_rate = learning_rate
        self.max_iters_in_epoch = max_iters_in_epoch
        self.grad_accumulate_steps = grad_accumulate_steps

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device('cuda:0' if torch.cuda.is_available()
                                       else 'cpu')

        self.epoch = 0

    def fit_dataset(self, data, collate_fn=None, callbacks=[]):
        # Start the training loop.
        while self.epoch < self.max_epochs:

            # train and evaluate train score
            print('training %i' % self.epoch)
            # TODO: create dataloader for `train`.
            # You can set batch_size as `self.batch_size` here,
            # and `collate_fn=collate_fn`.
            #print(data) #data is a dataset.DialogDataset object
            #dataloader = None
            dataloader = DataLoader(data, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)  
            # train epoch
            log_train = self._run_epoch(dataloader, True)
            #print("self.valid: ", self.valid) #this line will execute after one epoch of training is complete
            # evaluate valid score
            if self.valid is not None:
                print('evaluating %i' % self.epoch)
                # TODO: create dataloader for `valid`.
                # You can set batch_size as `self.batch_size` here,
                # and `collate_fn=collate_fn`.
                #is self.valid a dataset??? NOT so sure-->YES!
                #print("self.valid: ", self.valid)#self.valid IS A dataset.DialogDataset object
                # evaluate model
                #total # of samples in validation set:  5000
                dataloader = DataLoader(self.valid, batch_size=self.batch_size,shuffle=False ,collate_fn=collate_fn)
                log_valid = self._run_epoch(dataloader, False)
            else:
                log_valid = None

            for callback in callbacks:
                callback.on_epoch_end(log_train, log_valid, self)

            self.epoch += 1

    def predict_dataset(self, data,
                        collate_fn=default_collate,
                        batch_size=None,
                        predict_fn=None):
        if batch_size is None:
            batch_size = self.batch_size
        if predict_fn is None:
            predict_fn = self._predict_batch

        # set model to eval mode
        self.model.eval()

        # make dataloader
        # TODO: create dataloader for `data`.
        # You can set batch_size as `self.batch_size` here,
        # and `collate_fn=collate_fn`.
        # evaluate model
        #dataloader = None
        dataloader= DataLoader(data, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)
        ys_ = []
        with torch.no_grad():
            for batch in tqdm(dataloader):
                #print("batch: ", batch)
                #input()
                batch_y_ = predict_fn(batch)
                ys_.append(batch_y_)

        ys_ = torch.cat(ys_, 0)

        return ys_

    def save(self, path):
        torch.save({
            'epoch': self.epoch + 1,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        # TODO: Load saved model from path here.
        checkpoint= torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        epoch= checkpoint['epoch']
        #pass

    def _run_epoch(self, dataloader, training):
        # set model training/evaluation mode
        self.model.train(training)
        #print("dataloader:", dataloader) #torch.utils.data.dataloader.DataLoader object
        # run batches for train
        loss = 0

        # reset metric accumulators
        for metric in self.metrics:
            metric.reset()

        if training:
            iter_in_epoch = min(len(dataloader), self.max_iters_in_epoch)
            description = 'training'
        else: #evaluating
            iter_in_epoch = len(dataloader)
            description = 'evaluating'
        #print("len(dataloader): ", len(dataloader)) #10,000 for training; 500 for evaluation
        #there are 100,000 training data
        # 5000 validation data
        # run batches
        trange = tqdm(enumerate(dataloader),
                      total=iter_in_epoch,
                      desc=description)
        for i, batch in trange: #i means the i-th batch
            if training and i >= iter_in_epoch:
                break
            #print("batch: ", batch)
            #print("batch(options): ", batch["options"].shape)#torch.Size([10, 5, 50]) length of each option will change every batch
            #print("batch(context): ", batch["context"].shape) # torch.Size([10, 61]) length of each context will change every batch
            #print("batch(option_ids): ", len(batch["option_ids"])) #10
            if training:
                output, batch_loss = \
                    self._run_iter(batch, training)

                batch_loss /= self.grad_accumulate_steps

                # accumulate gradient - zero_grad
                #why do we need accumulate_steps?????
                if i % self.grad_accumulate_steps == 0:
                    # TODO: call zero gradient here
                    self.optimizer.zero_grad()      
                    #pass

                # TODO: Call backward on `batch_loss` here.
                batch_loss.backward()
                # accumulate gradient - step
                if (i + 1) % self.grad_accumulate_steps == 0:
                    # TODO: update gradient here
                    self.optimizer.step()
                    #pass
            else:
                with torch.no_grad():
                    output, batch_loss = \
                        self._run_iter(batch, training)
                    #print("output: ", output)
                    #print("batch: ", batch)

            # accumulate loss and metric scores
            loss += batch_loss.item()
            #print("loss:", loss)
            #print("output.shape: ", output.shape) #torch.Size([10, 5])
            #print("output: ", output)#
            #print("batch: ", batch) #no speaker, why???
            #input()
            for metric in self.metrics:
                #print("metric: ", metric) #metrics.Recall object
                metric.update(output, batch)
            #input()
            trange.set_postfix(
                loss=loss / (i + 1),
                **{m.name: m.print_score() for m in self.metrics})

        # calculate averate loss and metrics
        loss /= iter_in_epoch

        epoch_log = {}
        epoch_log['loss'] = float(loss)
        for metric in self.metrics:
            score = metric.get_score()
            print('{}: {} '.format(metric.name, score))
            epoch_log[metric.name] = score
        print('loss=%f\n' % loss)
        return epoch_log

    def _run_iter(self, batch, training):
        """ Run iteration for training.

        Args:
            batch (dict)
            training (bool)

        Returns:
            predicts: Prediction of the batch.
            loss (FloatTensor): Loss of the batch.
        """
        pass

    def _predict_batch(self, batch):
        """ Run iteration for predicting.

        Args:
            batch (dict)

        Returns:
            predicts: Prediction of the batch.
        """
        pass
