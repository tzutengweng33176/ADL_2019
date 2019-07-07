import argparse
import logging
import os
import pdb
import sys
import traceback
import pickle
import json
from embedding import Embedding
from preprocessor import Preprocessor


def main(args):

    preprocessor = Preprocessor(None)

    path, test_file = os.path.split(args.dest_file)
    #print(path)
    #print(test_file)

    # load embedding
    logging.info('loading embedding...')
    with open('./data/embedding.pkl', 'rb') as f:
        embedding = pickle.load(f)
        #config['model_parameters']['embedding'] = embedding.vectors
    
    preprocessor.embedding = embedding

    
    
    # test
    logging.info('Processing test from {}'.format(args.dest_file))
    test = preprocessor.get_dataset(
        args.dest_file, args.n_workers,
        {'n_positive': -1, 'n_negative': -1, 'shuffle': False}
    )
    test_pkl_path = os.path.join('./data/', 'test.pkl')
    logging.info('Saving test to {}'.format(test_pkl_path))
    with open(test_pkl_path, 'wb') as f:
        pickle.dump(test, f)


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess and generate preprocessed pickle.")
    parser.add_argument('dest_file', type=str,
                        help='[input] Path to the dest_file that .')
    parser.add_argument('--n_workers', type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s: %(message)s',
        level=logging.INFO, datefmt='%m-%d %H:%M:%S'
    )
    args = _parse_args()
    try:
        main(args)
    except KeyboardInterrupt:
        pass
    except BaseException:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
