# Recommender system with fairness regularization

This project implements recommender system with better fairness(D.E.E, [1]) metric in MovieLens1M. This code is based on the GLocal-K paper[2].


## Environment Installation

```bash
conda env create -f environment.yaml
conda activate tf1_env
```

```bash
cd data/MovieLens_1M
wget https://files.grouplens.org/datasets/movielens/ml-1m.zip
unzip ml-1m.zip
mv ml-1m/* ./
mv ratings.dat movielens_1m_dataset.dat
```


## Training

RMSE only training
```bash
python GLocal_K.py
```


RMSE + DEE training
```bash
python GLocal_K_refactor.py
```

To skip training, you can download checkpoints here: https://drive.google.com/drive/folders/1r0h8ps8ca4HrDHfFGOihxafcuyAuasyz?usp=sharing

## Inference

To infer your checkpoint
- ckpt after training GLocal_K_refactor.py -> move it to directory ```fair_ckpt```
- ckpt after training GLocal_K.py -> move it to directory ```accurate_ckpt```

RMSE only inference

```bash
python infer_accurate.py
```

RMSE + DEE inference

```bash
python infer_fair.py
```


## Result


Only RMSE (RMSE): 0.8237496

RMSE considering DEE: 0.83420455
DEE: 0.009363348093259893

Performance degrade: 1.26919%


[1] Jaewoong Cho, Moonseok Choi, & Changho Suh. (2022). Equal Experience in Recommender Systems.
[2] Han, S. C., Lim, T., Long, S., Burgstaller, B., & Poon, J. (2021, October). Glocal-k: Global and local kernels for recommender systems. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (pp. 3063-3067).