cd data/audio
wavpath='coraal/DCA_audio_2018.10.06'
mkdir -p ${wavpath}

for i in $(seq 10)
  do
    tar -xvf $(printf "DCA_audio_part%02d_2018.10.06.tar.gz" ${i}) --directory ${wavpath}
  done