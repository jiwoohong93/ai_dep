# 공정한 자원배분
자원배분 벤치마크 데이터셋인 NYC School 데이터를 사용하여 SAT 성적 평균을 최대화함과 동시에 공정성 조건을 만족하도록 인적 자원 배분을 하였다.

------------------------------

# 사용방법

+ 데이터셋 : school_data.csv

0. 패키지 설치
   pip install -r requirements.txt
1. CSIPM.py 실행 : final_data_interventions_max_frac.csv와 w_SIPM.npy 생성
   python3 codes/CSIPM.py
2. train_schools_SIPM_root.py 실행 : school_weights_linear_mse_sim5_max1_frac_sIPM_root_res.npz 생성
   python3 codes/train_schools_SIPM_root.py
3. intervention_sIPM_root.py 실행 : 자원배분 최적의 해를 찾고 결과를 csv 파일로 저장
   python3 codes/intervention_sIPM_root.py
