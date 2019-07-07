1. How to train my model?

Training data: First 1,000,000 sentences of the tokenized_corpus
Validation data: Last 5,000 sentences of the tokenized_corpus 
max_chars_len=16 
max_sent_len=64
batch_size=32
Epoch= 1
Optimizer= Adam
Learining rate= 0.001


python3 data_preprocessor.py \
	--train_path "/home/tzutengweng/ADLHW/A2/code_new/HW2/data/language_model/1M_corpus_tokenized.txt" \
	--valid_path "/home/tzutengweng/ADLHW/A2/code_new/HW2/data/language_model/valid_5k_corpus_tokenized.txt" \
	--model "test_1M_1_epoch_0422" \
	--batch_size=32 \
	--max_epoch=1

2. How to plot the figures in my report?
```
I used package matplotlib to plot my figure.

I used 2 lists all_train_ppl= [] , all_val_ppl=[]  to collect training perplexity and validation perlexity.

I appended training perplexity and validation perplexity every 1024 steps.


And I used the code below to plot my figure. 

plt.figure()
plt.plot(all_train_ppl)
plt.plot(all_val_ppl)
plt.savefig("train_and_val_ppl.png")
plt.show()


```
