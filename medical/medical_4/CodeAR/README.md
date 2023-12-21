# GAN-based Sequential Medical Data Generation

This repository contains code and instructions for generating sequential medical data using Generative Adversarial Networks (GAN) with the MIMIC-III dataset.

## Overview

The project uses a GAN model for data generation, implemented following the CodeAR model's architecture. For data representation, VQVAE is used within the discriminator. The generated data's effectiveness is evaluated using a Random Forest classifier as a baseline.

## Dataset

The MIMIC-III dataset is utilized for training the GAN model. Follow the instructions below to prepare the data:

1. Access the MIMIC-III dataset from [MLforHealth/MIMIC](https://github.com/MLforHealth/MIMIC).
2. Place the downloaded dataset into the `./Dataset` directory and preprocess it as instructed.

## Training

Training consists of the following steps:

1. Train the VQVAE model:
    ```bash
    python train_vqvae_step1.py --logdir=vqvae
    ```
2. Encode the dataset using the VQVAE model:
    ```bash
    python encoding_dataset.py --vqvae-dir=vqvae
    ```
3. Further train the VQVAE model:
    ```bash
    python train_vqvae_step2.py --vqvae-dir=vqvae --logdir=codear
    ```

## Data Generation

Generate synthetic data using the trained models:

1. Execute the `Data_generation.ipynb` notebook to produce synthetic data.

## Evaluation

Evaluate the synthetic data:

1. Use the `Classification.ipynb` notebook to assess the synthetic dataset by comparing the baseline classifier's performance on original and synthetic data.

Please follow the Jupyter notebooks provided for detailed steps on data generation and evaluation processes.
