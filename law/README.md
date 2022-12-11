1. 프로그램명: 법률 텍스트 내 편향성 제거기 (Debias Module on Legal Text)

2. 파일목록:
 (1) debias_legalText.py
  - Legal Text를 입력받아 언어 모델 내 편향성을 제거하기 위한 프로그램
  * debias_text
  - 입력) input_text: 편향성을 제거할 텍스트, label: 해당 텍스트에 대한 label, model_path: 학습된 모델을 load하기 위한 path
  - 출력) masked entities: 편향성이 포함된 단어, masked_text: 편향성이 제거된 텍스트

 (2) attribution.py
  - 모델의 출력 값 도출에 있어서 중요하게 작용한 단어를 색출하는 프로그램
  
  Class: ImportanceGetter
    * get_layers_importance
    - Explainability Method를 활용하여 모델의 입력 텍스트에 대한 단어 중요도 점수 산출
    * get_word_pieces_importance
    - get_layers_importance 함수를 통해 도출된 단어 별 중요도 점수를 토대로 모델의 예측에 중요하게 작용한 단어를 입력 텍스트 내에서 색출
    * cluster_word_importance
    - word-piece 단어들을 하나의 단어로 응집하고, 응집된 하나의 단어에 대한 중요도 점수를 계산

  (3) ner.py
  - 입력 텍스트 중 집단 및 사람에 해당하는 단어만을 색출하고, 해당 단어들을 masking하여 debiasing

  Class: NER
    * get_target_entities
    - ImportanceGetter의 함수들을 활용하여 도출된 중요 단어들에 한해서, 입력 텍스트 중 집단 및 사람에 해당하는 단어만을 색출
    * mask_original_text
    - get_target_entities 함수를 통해 도출된 집단 및 사람 단어들을 masking하여 Legal 텍스트를 debiasing