# sIPM-LFR: Learning fair representation with a parametric integral probability metric

Official pytorch implementation of ["Learning fair representation with a parametric integral probability metric"](https://arxiv.org/abs/2202.02943) published in [ICML 2022](https://icml.cc/Conferences/2022/) by Dongha Kim, Kunwoong Kim, Insung Kong, Ilsang Ohn, and Yongdai Kim.

## Dependencies

### Environments

- python 3.6+
- torch 1.11.0+
- CUDA 10.2+
- numpy 1.22.2+
- sklearn 1.1.0+
- argparse 1.1+
- yaml 6.0+

Automatically, those environmental dependencies are installed by running the following command:
```pip install -r requirements.txt```

Moreover, please make sure whether the CUDA environment is available.
This implementation is constructed over the GPU computing.

### Available datasets
- [Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult)
- [COMPAS dataset](https://github.com/propublica/compas-analysis)
- [Heritage Health dataset](https://foreverdata.org/1015/index.html)

Practitioners can freely use other custom datasets.

## Quick start

### Example commands

#### with a single fair hyperparameter
- For unsupervised LFR:
```python main.py --dataset adult --lmda 0.0 --lmdaR 1.0 --lmdaF 5.0 --head_net 1smooth```
- For supervised LFR:
```python main.py --dataset compas --lmda 1.0 --lmdaR 0.0 --lmdaF 0.1 --head_net 1smooth```

#### sweeping with many hyperparameters
- run ```./execute.bash``` for Adult dataset.

### Saved models and results
- The selected models and corresponding results are saved in folders ```/models``` and ```/results```.

### Citation
```
@InProceedings{kim2022sipmlfr,
  title = {Learning fair representation with a parametric integral probability metric},
  author = {Kim, Dongha and Kim, Kunwoong and Kong, Insung and Ohn, Ilsang and Kim, Yongdai},
  booktitle = {Proceedings of the 39th International Conference on Machine Learning},
  year = {2022}
}
```
