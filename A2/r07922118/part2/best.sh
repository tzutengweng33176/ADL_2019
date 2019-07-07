#!/bin/bash
python3.6 run_bert_predict.py --data_dir $(dirname $1) \
	--bert_model bert-large-uncased \
	--task_name ADLHW2 \
	--output_dir ./bert_large_uncased_output_3_epochs_b_32_seq_64_lr_2e_5_0421\
	--prediction_output_dir $(dirname $2)\
	--do_predict \
	--max_seq_length=64 \
  --do_lower_case \
	--cache_dir ./cache_dir

