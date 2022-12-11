# sweeping fair hyperparameters
fairs=(0.5 1.0 2.0 3.0 5.0 7.0 9.0 10.0 15.0 20.0 30.0 50.0 80.0 100.0)
for fair in "${fairs[@]}"
do
    python main.py --dataset health --lmda 0.0 --lmdaR 1.0 --lmdaF $fair --head_net 1smooth
done
