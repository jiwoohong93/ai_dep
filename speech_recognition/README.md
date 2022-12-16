# 음성 인식 모델

anaconda 가상환경 사용 권장

## 1. 사전 준비
1. speech_recognition 디렉토리로 이동 <br/>
2. ctcdecode 패키지 설치
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
cd ..
```
3. CORAAL 데이터셋 다운로드 및 압축해제
```
sh audio_download.sh 
sh merge_audio.sh
```
4. stft 추출
```
python extract_stft.py
```

## 2. 모델 학습
speech_recognition 디렉토리내에서
```
sh train.sh
```
혹은
```
python train.py --config_file {yml 파일이름} --model_name {모델이름} --lr {학습률} --batch_size {배치사이즈}
```

## 3. 모델 평가
모델 학습 후, speech_recognition 디렉토리내에서
```
sh evaluate.sh
```
혹은
```
python train.py --config_file {yml 파일이름} --model_name {모델이름} --checkpoint {모델파일(.pt) 경로} --eval
```

## 원본 코드
https://github.com/Hertin/Equal-Accuracy-Ratio
