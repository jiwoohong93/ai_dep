#!/bin/bash 
# subset = [chatdoctor_healthcare, chatdoctor_iclinieq]
# If public dataset is out-of-distribution, flag ood and specifiy ood_subset. 
python -u run.py \
      --data_name Malikeh1375/medical-question-answering-datasets \
      --ood \
      --ood_subset chatdoctor_healthcaremagic \
      --subset chatdoctor_icliniq  \
      --model_name gpt-3.5-turbo \
      --max_token 200 \
      --temp .8 \
      --ice_num 4 \
      --ds_size 100 \
      --ensemble 100 \
      --ojf output
