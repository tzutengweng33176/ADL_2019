#!/bin/bash
cd ELMo/test_1M_1_epoch_0422
wget -O char.dic https://www.dropbox.com/s/f9rs2outrqidabq/char.dic?dl=0
wget -O encoder.pkl https://www.dropbox.com/s/j85m51riurivd7q/encoder.pkl?dl=0
wget -O token_embedder.pkl https://www.dropbox.com/s/m4e4efcsveqa5xv/token_embedder.pkl?dl=0
wget -O word.dic https://www.dropbox.com/s/mzqwrx0hfppj57v/word.dic?dl=0
cd ../../model/submission/ckpts
wget -O epoch-7.ckpt https://www.dropbox.com/s/ksmr6np84okfhmy/epoch-7.ckpt?dl=0
cd ../../../

