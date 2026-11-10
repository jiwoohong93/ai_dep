# SLIDE

Official pytorch implementation of ["SLIDE: A surrogate fairness constraint to ensure fairness consistency"](https://www.sciencedirect.com/science/article/pii/S0893608022002891) published in [Neural Networks](https://www.journals.elsevier.com/neural-networks) (Volume 154, 2022, Pages 441-454) by Kunwoong Kim, Ilsang Ohn, Sara Kim, and Yongdai Kim.


## Usage

1. Locate your custom_dataset in the directory "datasets/{custom_dataset}".

2. Add loading function in "load.data_py" that should returns a tuple ```(xs, x, y, s)``` consisting four ``` torch.tensor ```
where ```xs = torch.cat([x, s.reshape(s.size(0), 1)], dim=1).```

3. run SLIDE as the command: "python main.py --dataset {custom_dataset} --lmda {lmda}"
where lmda is the fairness hyper-parameter, higher lmda increases the level of fairness (demographic parity or disparate impact).

For example, you can command
```python
python main.py --dataset law --lmda 5.0
```

## Environments

These codes are based on the following environments and versions of the corresponding libraries.

python >= 3.6
torch >= 1.8.0
