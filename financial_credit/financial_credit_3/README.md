#### Environment Setup
```
conda create -n finqa_roberta_v1 python=3.9
conda activate finqa_roberta_v1
pip install -r requirements.txt
```

##### Dataset & Model Setup
* Download: [drive_link](https://drive.google.com/file/d/1fvxWQ4MOkoDzF_zZE5-Zsp6ZiPaR6t1j/view?usp=drive_link)
* 압축 풀고 아래와 같이 배치해주시면 됩니다.
~~~~
├── finqa-roberta_v1.0
   ├── dataset
   ├── HuggingFaceCache
   ├── Readme.md
   ├── requirements.txt
   └── BERT_GBSQA.ipynb
~~~~    

##### Implementation
* 아래 파일 내 셀들을 실행해주시면 됩니다.
```
BERT_GBSQA.ipynb
```