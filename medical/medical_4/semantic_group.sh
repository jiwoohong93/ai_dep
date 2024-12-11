#!/bin/bash 
# --ood_path output/ood_medical-question-answering-datasets_chatdoctor_healthcaremagic_gpt-3.5-turbo_100way-4shot \

python semantic_group/semantic_group.py \
    --path output/medical-question-answering-datasets_chatdoctor_icliniq_gpt-3.5-turbo_100way-4shot \
    --embedding_model text-embedding-ada-002 \
    --query_type QA \
    --aggregate \
    --similarity_threshold 0.95 \
    --gamma 0.75  
