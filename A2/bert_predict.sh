#!/bin/bash 
python3 run_bert_predict.py --data_dir /home/tzutengweng/ADLHW/A2/code/HW2/data/classification \
	--bert_model bert-large-uncased \
	--task_name ADLHW2 \
	--output_dir ./bert_large_output_3_epochs_b_32_seq_64_lr_2e_5_0414\
	--do_predict \
	--max_seq_length=64 \
  --do_lower_case \
	--cache_dir ./cache_dir
