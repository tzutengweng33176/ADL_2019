#!/bin/bash
cd ./bert_large_uncased_output_3_epochs_b_32_seq_64_lr_2e_5_0421
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1E9BzPcUrwEnCsB2sw8ux-MT2RIQfti9V" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1E9BzPcUrwEnCsB2sw8ux-MT2RIQfti9V" -o bert_config.json
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=10EMGGlaAosKhNLtoirIRK_4psXSdw69o" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=10EMGGlaAosKhNLtoirIRK_4psXSdw69o" -o pytorch_model.bin
cd ../bert_uncased_output_3_epochs_b_32_seq_64_lr_2e_5_0421
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1MriaBu3dQKzoBls1vR08IynmtguZBrMf" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1MriaBu3dQKzoBls1vR08IynmtguZBrMf" -o bert_config.json
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=1snps_b8bRDDNr_x-MRMYbfGi8Pa_ynYq" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=1snps_b8bRDDNr_x-MRMYbfGi8Pa_ynYq" -o pytorch_model.bin
cd ..
