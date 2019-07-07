#!/bin/bash 
#mkdir -p model/ELMo_1M_test_1_epoch
#cp bcn_model_config_template.yaml model/ELMo_1M_test_1_epoch/config.yaml
python3 -m BCN.train model/ELMo_1M_test_1_epoch
