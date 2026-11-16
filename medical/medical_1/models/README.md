# Models

`src/smdart/train.py` 가 저장하는 model artifact 폴더입니다. 모델 파일은 저장소에 포함되어 있지 않습니다.

- `calibrated_xgboost_model.pkl`: 예측에 사용하는 최종 모델 (확률 보정 + 인과 개입 추론)
- `base_xgb_model.pkl`: 확률 보정 전 XGBoost 모델 (SHAP 분석용)
- `feature_names.pkl`: 추론 시 입력하는 설문 feature 순서
- `training_info.pkl`: 학습 메타데이터
