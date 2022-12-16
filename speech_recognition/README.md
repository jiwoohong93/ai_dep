유창동 교수님

# 음성 인식 모델


## 사전준비
1) speech_recognition 디렉토리로 이동
2) 터미널상에서 아래의 명령어 순차 실행
```
git clone --recursive https://github.com/parlance/ctcdecode.git
cd ctcdecode && pip install .
sh audio_download.sh 
sh merge_audio.sh
python extract_stft.py
```

## 모델 학습
speech_recognition 디렉토리내에서
```
train.sh
```
혹은
```
python train.py --config_file {yml 파일이름} --model_name {모델이름} --lr {학습률} --batch_size {배치사이즈}
```

## 원본 코드
https://github.com/Hertin/Equal-Accuracy-Ratio
