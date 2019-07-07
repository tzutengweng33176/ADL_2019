import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from metrics import Recall


def main(args):
    
    path, test_file = os.path.split(args.model_dir)
    
    path_to_csv, csv_file = os.path.split(args.dest_dir) 
    #load config
    config_path = os.path.join(path, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    # load embedding
    logging.info('loading embedding...')
    with open("./data/embedding.pkl", 'rb') as f:
        embedding = pickle.load(f)
        config['model_parameters']['embedding'] = embedding.vectors
    #print(embedding.word_dict)
    #word2index = embedding.word_dict
    #index2word =  {v: k for k, v in word2index.items()}
    #print(index2word)
    # make model
    if config['arch'] == 'ExampleNet':
        #from rnn_predictor import RNNPredictor
        from rnnatt_predictor import RNNAttPredictor
        #from best_predictor import BestRNNAttPredictor
        #PredictorClass =RNNPredictor
        PredictorClass= RNNAttPredictor
        #PredictorClass= BestRNNAttPredictor

    predictor = PredictorClass( metrics=[],
                               **config['model_parameters'])
    model_path = os.path.join(
        path,
        'model_rnnatt_0322.pkl.{}'.format(args.epoch))
    logging.info('loading model from {}'.format(model_path))
    predictor.load(model_path)

    # predict test
    logging.info('loading test data...')
    with open("./data/test.pkl", 'rb') as f:
        test = pickle.load(f)
        test.shuffle = False
    logging.info('predicting...')
    predicts = predictor.predict_dataset(test, test.collate_fn)
    #print("predicts: ", predicts.shape) #torch.Size([1000, 100]) 
    output_path = args.dest_dir 
    write_predict_csv(predicts, test, output_path)


def write_predict_csv(predicts, data, output_path, n=10):
    outputs = []
    for predict, sample in zip(predicts, data):
        #print("predict: ", predict)
        #print("predict.shape: ", predict.shape)# torch.Size([100])
        #print("sample: ", sample) # a dict
        candidate_ranking = [
            {
                'candidate-id': oid,
                'confidence': score.item()
            }
            for score, oid in zip(predict, sample['option_ids'])
        ] #a list of dicts
        #print("candidate_ranking: ", candidate_ranking[0])#{'candidate-id': '09XI5K556AP5', 'confidence': 0.2964949607849121}
        #print("len(candidate_ranking): ", len(candidate_ranking)) # 100
        #input()
        candidate_ranking = sorted(candidate_ranking,
                                   key=lambda x: -x['confidence']) #sort by decreasing confidence
        best_ids = [candidate_ranking[i]['candidate-id']
                    for i in range(n)]    #take top-n as best-ids
        outputs.append(
            ''.join(
                ['1-' if oid in best_ids
                 else '0-'
                 for oid in sample['option_ids']])
        )

    logging.info('Writing output to {}'.format(output_path))
    with open(output_path, 'w') as f:
        f.write('Id,Predict\n')
        for output, sample in zip(outputs, data):
            f.write(
                '{},{}\n'.format(
                    sample['id'],
                    output
                )
            )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('dest_dir', type=str,
                        help='Directory to the predicted csv.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--not_load', action='store_true',
                        help='Do not load any model.')
    parser.add_argument('--epoch', type=int, default=9)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = _parse_args()
    logging.basicConfig(format='%(asctime)s | %(levelname)s | %(message)s',
                        level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
