## Dataset (SDM-ART) 
데이터: 분당서울대학교병원의 만성콩팥병 환자 투석방법 선택을 위한 공동의사결정 임상시험

*실제 의료 데이터로, 취득을 위해선 분당서울대학교병원과 DRB, IRB 절차 필요*

본 연구에서 사용된 데이터는 분당서울대학교병원의 실제 데이터로, 만성콩팥병 환자 투석방법 선택을 위한 공동의사결정 임상시험을 통해 수집된 약 700명 이상의 환자 데이터를 기반으로 한다. 데이터는 다음과 같은 정보를 포함한 tabular 데이터셋이다:
- 인구통계학적 정보: 환자의 성별, 나이, 사회적 신분 등
- 의학적 검사 정보: 혈액 검사, 이학적 검사, 흉부 방사선, 심전도 등
- 설문조사 문항: 약 150개로 구성된 치료법 교육 및 자가진단 설문조사 문항(질병 인식, 자가 관리, 환자 및 의사 만족도 등)
환자들에게 제공된 치료법은 다음의 네 가지로 구성된다: 혈액투석, 복막투석, 신장이식, 치료법 선택 안 함. 또한, 환자의 응급투석 기록도 함께 수집되었다.
환자들은 각 치료법에 대한 상세 정보를 치료법 선택 전까지 주치의를 통해 최대 7회까지 교육받을 수 있었으며, 교육 후 치료법을 선택한다면 그 이후의 교육 데이터는 없다. 
치료법 선택 후의 환자 만족도 관련 설문조사가 추가로 수집되었으며, 치료법 교육을 받은 그룹과 받지 않은 그룹이 무작위로 배정되어 교육의 효과를 비교 가능하다.

# MediSurveyBoost
입력: 환자의 screening 정보 (인구통계학적 정보, 활력징후, 가족 및 병력 조사, 흉부 방사선 및 심전도, 이학적 검사, 혈액학적 검사, 혈액화학적 검사 등) 및 첫 교육 후 설문조사 문항 (교육 후 자가진단 도구, 질병 인식, 환자 만족도 및 순응도, 의사 만족도, 삶의 질 등)
출력: 교육 후 응급 투석 여부 (이진 분류)

본 연구에서는 상대적으로 적은 데이터에서도 강건한 성능을 발휘하고, feature importance 분석이 용이한 XGBoost classifier를 기반으로 학습을 수행했다. 
초기 학습 결과를 바탕으로 결정에 큰 영향을 미치는 주요 feature를 분석한 결과, 의학적 검사 정보 외에도 가족 관계 관련 항목과 자가 관리 관련 설문조사 문항이 상위권에 나타났다.
자가 관리가 소홀한 것은 응급 투석 위험과 직접적으로 연관될 수 있는 feature로 이해되었으나, 가족 관계와 같은 인구통계학적 정보는 과도한 영향을 미칠 경우 편향성을 초래할 가능성이 있다고 판단되었다. 이에 따라, 가족 관계 관련 설문 문항을 중점으로 편향성 완화 기법을 도입하였다.
편향성 완화 기법으로는 Inference 단계에서의 인과 개입(Causal Intervention)을 활용하였으며, 이는 do-operator를 통해 구현되었다. 이 기법은 특정 feature(가족 관계 관련 문항)가 모델 판단에 미치는 영향을 통제하여, 그 외 일반 변수의 인과적 영향만을 모델이 올바르게 인지하도록 유도한다. 구체적으로는, 모델 입력 데이터에서 가족 관계 관련 문항의 답변을 통제하여 해당 문항에 대해 다른 답변을 가정한 모든 가능성의 입력 데이터를 생성하여 각 입력 데이터에 대해 추론을 수행한 결과값을 도출했다. 그 후 학습 데이터에서의 해당 문항의 답변 분포(1~5) 통계치를 기반으로, 생성된 모든 가능성의 추론 결과를 가중 평균하여 최종 예측값을 산출하였다. 이 접근법은 가족 관계 관련 문항의 편향된 영향력을 줄이고, 모델이 환자의 의학적 상태와 자가 관리와 같은 본질적 요인에 근거하여 예측을 수행하도록 유도한다. 


## Project Structure

```
.
├── SDM-ART
│   └── preprocess.py
├── analysis.py
├── dataset.py
├── environment.yaml
├── medisurveyboost.py
```

## Environment

```
conda create -n medisurveyboost python=3.8
conda activate medisurveyboost
pip install -r requirements.txt3
```

## Usage

```
cd SDM-ART
python preprocess.py
cd ..
python medisurveyboost.py
python analysis.py
```

## Results

|                   | 단위 | 성능 |
| :---------------: | :---------------------: | :-------------: |
| 의료 알고리즘 정확도  |          %           |      76.56      |
| 의료 알고리즘 공정성 |        개별공정성(%)         |    85.74    |
