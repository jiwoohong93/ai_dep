#!/usr/bin/env python
# coding: utf-8

# In[1]:


from time import time
from scipy.sparse import csc_matrix
import tensorflow as tf
import numpy as np
import h5py
import random
from regularizers import FairnessLoss
from additional_load import data_loader_movielens_additional


seed = 42
random.seed(seed)
np.random.seed(seed)
tf.set_random_seed(seed)
scale_loss = 1.0
LR = 0.001

# In[3]:

def load_data_1m(path='./', delimiter='::', frac=0.1, seed=1234):

    tic = time()
    print('reading data...')
    data = np.loadtxt(path+'movielens_1m_dataset.dat', skiprows=0, delimiter=delimiter).astype('int32')
    print('taken', time() - tic, 'seconds')

    n_u = np.unique(data[:,0]).size  # num of users
    n_m = np.unique(data[:,1]).size  # num of movies
    n_r = data.shape[0]  # num of ratings

    udict = {}
    for i, u in enumerate(np.unique(data[:,0]).tolist()):
        udict[u] = i
    mdict = {}
    for i, m in enumerate(np.unique(data[:,1]).tolist()):
        mdict[m] = i

    np.random.seed(seed)
    idx = np.arange(n_r)
    np.random.shuffle(idx)

    train_r = np.zeros((n_m, n_u), dtype='float32')
    test_r = np.zeros((n_m, n_u), dtype='float32')

    for i in range(n_r):
        u_id = data[idx[i], 0]
        m_id = data[idx[i], 1]
        r = data[idx[i], 2]

        if i < int(frac * n_r):
            test_r[mdict[m_id], udict[u_id]] = r
        else:
            train_r[mdict[m_id], udict[u_id]] = r

    train_m = np.greater(train_r, 1e-12).astype('float32')  # masks indicating non-zero entries
    test_m = np.greater(test_r, 1e-12).astype('float32')

    print('data matrix loaded')
    print('num of users: {}'.format(n_u))
    print('num of movies: {}'.format(n_m))
    print('num of training ratings: {}'.format(n_r - int(frac * n_r)))
    print('num of test ratings: {}'.format(int(frac * n_r)))

    return n_m, n_u, train_r, train_m, test_r, test_m, udict, mdict


# # Load Data

# In[4]:

# Insert the path of a data directory by yourself (e.g., '/content/.../data')
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
data_path = './data'
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
# Select a dataset among 'ML-1M', 'ML-100K', and 'Douban'
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
dataset = 'ML-1M'
# .-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._.-^-._
path = data_path + '/MovieLens_1M/'
n_m, n_u, train_r, train_m, test_r, test_m, udict, mdict = load_data_1m(path=path, delimiter='::', frac=0.1, seed=1234)
user, item = data_loader_movielens_additional()
for k in user.keys():
    user[k] = [udict[u+1] for u in user[k]]
for k in item.keys():
    tmp = []
    for i in range(len(item[k])):
        if item[k][i]+1 not in mdict.keys():
            continue
        tmp.append(mdict[item[k][i]+1])
    item[k] = tmp

# In[5]:


print(np.where(test_m == 1))


# # Hyperparameter Settings

# In[6]:


# Common hyperparameter settings
n_hid = 500
n_dim = 5
n_layers = 2
gk_size = 3


# In[7]:


lambda_2 = 70.
lambda_s = 0.018
iter_p = 50
iter_f = 10
epoch_p = 600
epoch_f = 5000
dot_scale = 0.5


# Additional Parameters

# In[8]:


lambda_fair = 0.9
tau=3


# In[9]:


R = tf.placeholder("float", [n_m, n_u])
male_user_idx = tf.placeholder("int32", [None])
male_item_idx = tf.placeholder("int32", [None])
female_user_idx = tf.placeholder("int32", [None])
female_item_idx = tf.placeholder("int32", [None])


f_criterion = FairnessLoss(h=0.01, tau=tau, delta=0.01, male_user_idx=male_user_idx, female_user_idx=female_user_idx, male_item_idx=male_item_idx, female_item_idx=female_item_idx)


# # Network Function

# In[10]:


def local_kernel(u, v):

    dist = tf.norm(u - v, ord=2, axis=2)
    hat = tf.maximum(0., 1. - dist**2)

    return hat


# In[11]:


def kernel_layer(x, n_hid=n_hid, n_dim=n_dim, activation=tf.nn.sigmoid, lambda_s=lambda_s, lambda_2=lambda_2, name=''):

    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        W = tf.get_variable('W', [x.shape[1], n_hid])
        n_in = x.get_shape().as_list()[1]
        u = tf.get_variable('u', initializer=tf.random.truncated_normal([n_in, 1, n_dim], 0., 1e-3))
        v = tf.get_variable('v', initializer=tf.random.truncated_normal([1, n_hid, n_dim], 0., 1e-3))
        b = tf.get_variable('b', [n_hid])

    w_hat = local_kernel(u, v)
    
    sparse_reg = tf.contrib.layers.l2_regularizer(lambda_s)
    sparse_reg_term = tf.contrib.layers.apply_regularization(sparse_reg, [w_hat])
    
    l2_reg = tf.contrib.layers.l2_regularizer(lambda_2)
    l2_reg_term = tf.contrib.layers.apply_regularization(l2_reg, [W])

    W_eff = W * w_hat  # Local kernelised weight matrix
    y = tf.matmul(x, W_eff) + b
    y = activation(y)

    return y, sparse_reg_term + l2_reg_term


# In[12]:


def global_kernel(input, gk_size, dot_scale):

    avg_pooling = tf.reduce_mean(input, axis=1)  # Item (axis=1) based average pooling
    avg_pooling = tf.reshape(avg_pooling, [1, -1])
    n_kernel = avg_pooling.shape[1].value

    conv_kernel = tf.get_variable('conv_kernel', initializer=tf.random.truncated_normal([n_kernel, gk_size**2], stddev=0.1))
    gk = tf.matmul(avg_pooling, conv_kernel) * dot_scale  # Scaled dot product
    gk = tf.reshape(gk, [gk_size, gk_size, 1, 1])

    return gk


# In[13]:


def global_conv(input, W):

    input = tf.reshape(input, [1, input.shape[0], input.shape[1], 1])
    conv2d = tf.nn.relu(tf.nn.conv2d(input, W, strides=[1,1,1,1], padding='SAME'))

    return tf.reshape(conv2d, [conv2d.shape[1], conv2d.shape[2]])


# # Network Instantiation

# ## Pre-training

# In[14]:


y = R
reg_losses = None

for i in range(n_layers):
    y, reg_loss = kernel_layer(y, name=str(i))
    reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss

pred_p, reg_loss = kernel_layer(y, n_u, activation=tf.identity, name='out')
reg_losses = reg_losses + reg_loss

# L2 loss
diff = train_m * (train_r - pred_p)
sqE = tf.nn.l2_loss(diff)
loss_p_tmp = (sqE + reg_losses) / tf.reduce_sum(train_m)
if f_criterion is not None:
    fair_lossp = f_criterion(tf.matrix_transpose(pred_p))
    loss_p = (1. - lambda_fair) * loss_p_tmp + lambda_fair * fair_lossp
else:
    loss_p = loss_p_tmp

loss_p *= scale_loss

optimizer_p = tf.contrib.opt.ScipyOptimizerInterface(loss_p, options={'disp': True, 'maxiter': iter_p, 'maxcor': 10, 'ftol':1e-8}, method='L-BFGS-B')


# ## Fine-tuning

# In[15]:


y = R
reg_losses = None

for i in range(n_layers):
    y, _ = kernel_layer(y, name=str(i))

y_dash, _ = kernel_layer(y, n_u, activation=tf.identity, name='out')

gk = global_kernel(y_dash, gk_size, dot_scale)  # Global kernel
y_hat = global_conv(train_r, gk)  # Global kernel-based rating matrix

for i in range(n_layers):
    y_hat, reg_loss = kernel_layer(y_hat, name=str(i))
    reg_losses = reg_loss if reg_losses is None else reg_losses + reg_loss

pred_f, reg_loss = kernel_layer(y_hat, n_u, activation=tf.identity, name='out')
reg_losses = reg_losses + reg_loss

# L2 loss
diff = train_m * (train_r - pred_f)
sqE = tf.nn.l2_loss(diff)
loss_f = (sqE + reg_losses) / tf.reduce_sum(train_m)

if f_criterion is not None:
    fair_lossf = f_criterion(tf.matrix_transpose(pred_f))
    loss_f = (1. - lambda_fair) * loss_f + lambda_fair * fair_lossf

loss_f *= scale_loss

optimizer_f = tf.contrib.opt.ScipyOptimizerInterface(loss_f, options={'disp': True, 'maxiter': iter_f, 'maxcor': 10, 'ftol':1e-8}, method='L-BFGS-B')
#adam_f = tf.train.AdamOptimizer(learning_rate=LR)




# # Evaluation code

# In[16]:


def dcg_k(score_label, k):
    dcg, i = 0., 0
    for s in score_label:
        if i < k:
            dcg += (2**s[1]-1) / np.log2(2+i)
            i += 1
    return dcg


# In[17]:


def ndcg_k(y_hat, y, k):
    score_label = np.stack([y_hat, y], axis=1).tolist()
    score_label = sorted(score_label, key=lambda d:d[0], reverse=True)
    score_label_ = sorted(score_label, key=lambda d:d[1], reverse=True)
    norm, i = 0., 0
    for s in score_label_:
        if i < k:
            norm += (2**s[1]-1) / np.log2(2+i)
            i += 1
    dcg = dcg_k(score_label, k)
    return dcg / norm


# In[18]:


def call_ndcg(y_hat, y):
    ndcg_sum, num = 0, 0
    y_hat, y = y_hat.T, y.T
    n_users = y.shape[0]

    for i in range(n_users):
        y_hat_i = y_hat[i][np.where(y[i])]
        y_i = y[i][np.where(y[i])]

        if y_i.shape[0] < 2:
            continue

        ndcg_sum += ndcg_k(y_hat_i, y_i, y_i.shape[0])  # user-wise calculation
        num += 1

    return ndcg_sum / num


# # Training and Test Loop

# In[19]:


best_rmse_ep, best_mae_ep, best_ndcg_ep = 0, 0, 0
best_rmse, best_mae, best_ndcg = float("inf"), float("inf"), 0

time_cumulative = 0
init = tf.global_variables_initializer()

saver = tf.train.Saver()


def get_dee(pre):
    DEE = 0
    pred_hat = np.where(pre > tau, 1, 0)
    for g in ['M', 'F']:
        for i in ['M', 'F']:
            DEE += np.abs(np.mean(pred_hat)-np.mean(pred_hat[user[g]][:, item[i]]))
    
    return DEE

with tf.Session() as sess:
    # load ckpt
    saver.restore(sess, './accurate_ckpt/ML-1M_last_rmse.ckpt')
    cur_feed = {R:test_r}
    pre = sess.run(pred_f, feed_dict=cur_feed)
    error = (test_m * (np.clip(pre, 1., 5.) - test_r) ** 2).sum() / test_m.sum()  # test error
    test_rmse = np.sqrt(error)
    print('test rmse:', test_rmse)