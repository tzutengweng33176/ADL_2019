#!/bin/bash 
python3 run_classifier.py \
	--data_dir /home/tzutengweng/ADLHW/A2/code/HW2/data/classification \
	--bert_model bert-large-cased\
	--task_name ADLHW2 \
	--output_dir ./bert_large_cased_output_3_epochs_b_32_seq_64_lr_2e_5_0414\
	--do_eval \
	--do_train \
	--train_batch_size=32  \
	--max_seq_length=64 \
	--num_train_epochs 3 \
	--learning_rate 2e-5 \
	--gradient_accumulation_steps=10 \
	--cache_dir ./cache_dir
