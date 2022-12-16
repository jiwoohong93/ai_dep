유창동 교수님

음성 인식 모델

# 사전준비
1) speech_recognition 디렉토리로 이동
2) 디렉토리 내 ctcdecode 패키지 설치 (https://github.com/parlance/ctcdecode 참조)
3) 아래의 쉘 스크립트 순서대로 실행
$ sh audio_download.sh
$ sh merge_audio.sh
4) 다음을 실행
$ python extract_stft.py

# 모델 학습
$ train.sh


원본 코드: https://github.com/Hertin/Equal-Accuracy-Ratio
