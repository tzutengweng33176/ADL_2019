#!/bin/bash
python3 data_preprocessor.py \
	--train_path "/home/tzutengweng/ADLHW/A2/code_new/HW2/data/language_model/1M_corpus_tokenized.txt" \
	--valid_path "/home/tzutengweng/ADLHW/A2/code_new/HW2/data/language_model/valid_5k_corpus_tokenized.txt" \
	--model "test_1M_1_epoch_0422" \
	--batch_size=32 \
	--max_epoch=1

