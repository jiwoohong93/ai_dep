import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs

from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from aif360.datasets import BinaryLabelDataset
import time
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from sklearn.decomposition import TruncatedSVD
from torch.utils.data import TensorDataset, DataLoader


""" 2 Gaussian toy dataset """

def load_toy_dataset(n = 10000, dim = 20) :
    
    xs_train_sub, y_train = make_blobs(n_samples = n, n_features = dim, centers = 2)
    xs_test_sub, y_test = make_blobs(n_samples = int(n/2), n_features = dim, centers = 2)
    
    x_train, x_test = xs_train_sub[:, :-1], xs_test_sub[:, :-1]
    s_train, s_test = xs_train_sub[:, -1], xs_test_sub[:, -1]
    
    # toy sensitives
    s_train, s_test = (s_train > 0) * 1, (s_test > 0) * 1
    
    xs_train = np.hstack((x_train, s_train.reshape(s_train.shape[0], 1)))
    xs_test = np.hstack((x_test, s_test.reshape(s_test.shape[0], 1)))
   
    return (xs_train, x_train, y_train, s_train), (xs_test, x_test, y_test, s_test)



""" Law school dataset """

def load_law_dataset(seed = 2021, testsize = 0.20) :

    df = pd.read_csv('datasets/law/law_data.csv', index_col=0)
    Y = np.array([int(y == "Passed") for y in df["pass_bar"]])
    Z = np.array([int(z == "White") for z in df["race"]])
    col_quanti = ['zfygpa', 'zgpa', 'DOB_yr', 'cluster_tier', 'family_income',
            'lsat', 'ugpa', 'weighted_lsat_ugpa']
    col_quali = ['isPartTime', 'sex']

    X_quali = df[col_quali].values
    X_quanti = df[col_quanti].values

    quali_encoder = OneHotEncoder(categories="auto", drop = 'first')
    quali_encoder.fit(X_quali)

    X_quali = quali_encoder.transform(X_quali).toarray()

    X = np.concatenate([X_quanti, X_quali], axis=1)

    y0_idx = np.where(Y==0)[0]
    y1_idx = np.where(Y==1)[0]

    y0_train_idx, y0_test_idx = train_test_split(y0_idx, test_size=testsize, random_state=seed)
    y1_train_idx, y1_test_idx = train_test_split(y1_idx, test_size=testsize, random_state=seed)

    train_idx = np.concatenate((y0_train_idx, y1_train_idx))                                
    test_idx = np.concatenate((y0_test_idx, y1_test_idx))

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    Z_train = Z[train_idx]

    X_test = X[test_idx]
    Y_test = Y[test_idx]
    Z_test = Z[test_idx]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    XZ_train = np.concatenate([X_train, Z_train.reshape(Z_train.shape[0], 1)], axis = 1)
    XZ_test = np.concatenate([X_test, Z_test.reshape(Z_test.shape[0], 1)], axis = 1)

    return (XZ_train, X_train, Y_train, Z_train), (XZ_test, X_test, Y_test, Z_test)



def flip_sen_datasets(XS) :

    sen_idx = XS.shape[1] - 1

    XS_first = XS.clone()
    XS_first[:, sen_idx] = 1

    XS_second = XS.clone()
    XS_second[:, sen_idx] = 0

    first_set, second_set = TensorDataset(XS_first), TensorDataset(XS_second)

    return first_set, second_set, XS_first, XS_second


