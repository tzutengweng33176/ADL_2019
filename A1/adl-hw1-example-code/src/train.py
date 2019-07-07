import argparse
import logging
import os
import pdb
import pickle
import sys
import traceback
import json
from callbacks import ModelCheckpoint, MetricsLogger
from metrics import Recall


def main(args):
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path) as f:
        config = json.load(f)

    logging.info('loading embedding...')
    with open(config['model_parameters']['embedding'], 'rb') as f:
        embedding = pickle.load(f)
        config['model_parameters']['embedding'] = embedding.vectors

    logging.info('loading valid data...')
    with open(config['model_parameters']['valid'], 'rb') as f:
        config['model_parameters']['valid'] = pickle.load(f)

    logging.info('loading train data...')
    with open(config['train'], 'rb') as f:
        train = pickle.load(f)
    #print(train) #dataset.DialogDataset object
    #word2index = embedding.word_dict
    #index2word =  {v: k for k, v in word2index.items()}

    if config['arch'] == 'ExampleNet':
        #from example_predictor import ExamplePredictor
        #from rnn_predictor import RNNPredictor
        #from best_predictor import BestRNNAttPredictor
        from rnnatt_predictor import RNNAttPredictor
        #PredictorClass = ExamplePredictor
        #PredictorClass = RNNPredictor
        PredictorClass = RNNAttPredictor
        #PredictorClass = BestRNNAttPredictor

    #print("config['model_parameters']: ", config['model_parameters']) #it's a dict; {'valid': dataset.DialogDataset object, 'embedding':a big tensor}
    #print("**config['model_parameters']: ", **config['model_parameters'])
    predictor = PredictorClass(
        metrics=[Recall()],
        **config['model_parameters']
    )
    # **dict : https://stackoverflow.com/questions/21809112/what-does-tuple-and-dict-means-in-python
    #input()
    if args.load is not None:
        predictor.load(args.load)

    model_checkpoint = ModelCheckpoint(
        os.path.join(args.model_dir, 'model_rnnatt_6_negative_samples_0324.pkl'),
        'loss', 1, 'all'
    )
    metrics_logger = MetricsLogger(
        os.path.join(args.model_dir, 'log_rnnatt_6_neagtive_samples_0324.json')
    )

    logging.info('start training!')
    predictor.fit_dataset(train,
                          train.collate_fn,
                          [model_checkpoint, metrics_logger])


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train.")
    parser.add_argument('model_dir', type=str,
                        help='Directory to the model checkpoint.')
    parser.add_argument('--device', default=None,
                        help='Device used to train. Can be cpu or cuda:0,'
                        ' cuda:1, etc.')
    parser.add_argument('--load', default=None, type=str)
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
