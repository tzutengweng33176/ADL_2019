import torch


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Recall(Metrics):
    """
    Args:
         ats (int): @ to eval.
         rank_na (bool): whether to consider no answer.
    """
    def __init__(self, at=10):
        self.at = at
        self.n = 0
        self.n_correct = 0
        self.name = 'Recall@{}'.format(at)

    def reset(self):
        self.n = 0
        self.n_corrects = 0

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        predicts = predicts.cpu() #tensor.cuda() is used to move a tensor to GPU memory.
                                  #tensor.cpu() moves it back to memory accessible to the CPU.
        #print("predicts: ", predicts.shape) #still a tensor; torch.Size([10, 5]) when training; torch.Size([10, 100]) when validating
        # TODO
        # This method will be called for each batch.
        # You need to
        # - increase self.n, which implies the total number of samples.
        #ONLY positive samples can be considered as relevant items
        #if that positive samples is NOT in top 10, then it will be considered as false negative
        self.n+= predicts.shape[0]
        # - increase self.n_corrects based on the prediction and labels
        #   of the batch
        for i in range(predicts.shape[0]): #predicts.shape[0] is the batch size 
            correct_id = batch['option_ids'][i][0] #correct id of the i-th sample
            #print("correct_id: ", correct_id)
            #print("predicst[i]: ", predicts[i])
            #print("batch['option_ids'][i]: ", batch['option_ids'][i])
            #for score, oid in zip(predicts[i], batch['option_ids'][i]):
            candidate_ranking = [
                {
                    'candidate-id': oid, 
                    'confidence': score.item()
                        
                }
                for score, oid in zip(predicts[i], batch['option_ids'][i])
                    ]
            candidate_ranking = sorted(candidate_ranking,
                                       key=lambda x: -x['confidence'])    
            #print("candidate_ranking: ", candidate_ranking)
            if(predicts.shape[1]==100):
                best_ids = [candidate_ranking[j]['candidate-id']
                            for j in range(self.at)]    #take top-n as best-ids
                #print("best_ids: ", best_ids)
            else:
                best_ids = [candidate_ranking[k]['candidate-id']
                            for k in range(5)  
                        ]
                #print("best_ids: ", best_ids) #len(best_ids)=5 
        # what is the meaning of number in predicts? confidence score-->if the confidence score is higher, the more likely it will be the correct answer
        #so we should sort the scores in predicts in descreasing order and find the corresponding id of each score
        #if the corresponding id of the highest score matches the label, then it's True positivie
        #recall @ 10 should be evaluated on testing or validation set (?); when in training set
        # https://surprise.readthedocs.io/en/latest/FAQ.html#how-to-compute-precision-k-and-recall-k
        #there is also a correct answer for each sample in validation set
            if correct_id in best_ids:
                self.n_corrects+=1

    def get_score(self):
        #print("self.n_corrects: ", self.n_corrects)
        #print("self.n: ", self.n)
        return self.n_corrects / self.n

    def print_score(self):
        score = self.get_score()
        return '{:.2f}'.format(score)
