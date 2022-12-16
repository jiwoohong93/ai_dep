유창동 교수님

# 음성 인식 모델
'''
echo 1;
'''


## 사전준비
1) speech_recognition 디렉토리로 이동
2) 디렉토리 내 ctcdecode 패키지 설치 (https://github.com/parlance/ctcdecode 참조)
3) 아래의 쉘 스크립트 순서대로 실행 <br/>
$ sh audio_download.sh </br>
$ sh merge_audio.sh
4) 다음을 실행 <br/>
$ python extract_stft.py

## 모델 학습
1) 다음 쉘 스크립트를 실행하거나 <br/>
$ train.sh
2) 명령어 직접 입력 <br/>
$ python train.py --config_file {yml 파일이름} --model_name {모델이름} --lr {학습률} --batch_size {배치사이즈}


원본 코드: https://github.com/Hertin/Equal-Accuracy-Ratio
