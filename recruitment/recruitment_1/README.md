목표:

기업을 통해 확보한 실데이터인 [기업정보], [공고정보], [이력정보] 데이터를 통해서 개인과 기업을 연결시켜주는 추천 인공지능 모델을 구축. 

불필요한 데이터(모집인원 0명)와 비어있는 데이터(자본금,매출현황,사원수가 0이거나 공고 키워드 없음)에 대한 전처리.



[기업정보]와 [공고정보]를 '기업 번호'에 따라 매칭시켜 [기업공고정보]와 [이력정보] 2개의 데이터로 분리.



서로에 대한 적합성 점수를 부여하는 회귀(Regression) 문제로 접근.

[이력정보] 내의 과거 경력을 기반으로 적합성을 논의.


