DOWNLOAD pre-trained model using "bash download.sh" and then you can predict

1. How to train my model?

#!/bin/bash 
python3 run_classifier.py \
	--data_dir /path/to/test.csv \ 
	--bert_model bert-base-cased\
	--task_name ADLHW2 \
	--output_dir /path/to/output \
	--do_eval \
	--do_train \
	--train_batch_size=32  \
	--max_seq_length=64 \
	--num_train_epochs 3 \
	--learning_rate 2e-5 \
	--gradient_accumulation_steps=10 \
	--cache_dir ./cache_dir

After training, you will get pytorch.bin model file in the output directory.


