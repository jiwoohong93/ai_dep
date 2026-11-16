# Data

## Dataset 구조

```
data/
│
├── original/
│   ├── image/
│   └── text/
├── test/
│   ├── image/
│   └── text/
│
└── README.md
```

- original : 원본데이터 저장소
- test : 테스트 샘플 저장소
- text/image : 모델에게 학습하는 데이터에 따라 분리(재무제표 추출값 / 재무제표 원본)
