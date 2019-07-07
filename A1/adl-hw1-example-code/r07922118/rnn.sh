#!/bin/bash
python3.7 make_dataset.py $1
python3.7 predict.py ./models/ $2 --epoch 9
