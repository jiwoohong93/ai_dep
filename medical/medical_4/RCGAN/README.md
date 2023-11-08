# GAN-based Sequntial Medical Data Generation
In this repository, it is possible to train a GAN-based sequential medical data generation model based on [eICU dataset](https://eicu-crd.mit.edu/).
The GAN model is implemented based on [RCGAN](https://arxiv.org/abs/1706.02633v2) architecture and the discriminator is implemented following [LSGAN](https://openaccess.thecvf.com/content_iccv_2017/html/Mao_Least_Squares_Generative_ICCV_2017_paper.html) loss function instead of the vanila GAN loss function, which is used in the original RCGAN model.
In addition, after training GAN models, it is possible to synthetically generate the medical data and evaluate its effectiveness based on baseline medical data classifier.

## Dataset Download
1. Download the medical [eICU dataset](https://eicu-crd.mit.edu/) and locate the folder in the Dataset directory
2. Do preprocessing of the dataset using `Preprocessing.ipynb`
3. Here, the labels are also extracted following [RCGAN](https://arxiv.org/abs/1706.02633v2) to train medical data classifiers, where they are trained to predict whether a value of a specific measurement is higher or smaller than a pre-defined threshold based on the mesurement values over the previous 4 hours

## Train and Evaluate a GAN model generating sequential medical data
1. Train your GAN models using `train_gan.py`
```python
python train_gan.py --gpu=0 --logdir="baseline"  
```  
2. Using `Data_generation.ipynb`, it is possible to generate the synthetic dataset using the trained GAN models
3. Using `Classification.ipynb`, it is possible to evaluate the synthetic dataset by comparing the accuracies of the baseline medical data classifier trained on original dataset or synthetic dataset
