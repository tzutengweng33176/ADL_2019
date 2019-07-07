#!/bin/sh
cd data
wget -O embedding.pkl https://www.dropbox.com/s/wpxben18wgbr5jo/embedding.pkl?dl=0
cd ../models
wget -O model_best_0324.pkl.9 https://www.dropbox.com/s/sc0q0vrxcgq6ubm/model_best_0324.pkl.9?dl=0
wget -O model_rnn_0323.pkl.9 https://www.dropbox.com/s/l9o6gbntx446bx0/model_rnn_0323.pkl.9?dl=0
wget -O model_rnnatt_0322.pkl.8 https://www.dropbox.com/s/sqd2tpcumosbh86/model_rnnatt_0322.pkl.8?dl=0

