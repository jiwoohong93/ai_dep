import tensorflow as tf
import numpy as np
import math

# Disable TensorFlow v2 behavior
#tf.disable_v2_behavior()


def normal_pdf(x):
    return tf.exp(-0.5 * tf.square(x)) / math.sqrt(2 * math.pi)


def normal_cdf(y, h=0.01, tau=0.5):
    Q_fn = lambda x: tf.exp(-0.4920 * tf.square(x) - 0.2887 * x - 1.1893)
    y_prime = (tau - y) / h
    positive_mask = tf.greater(y_prime, 0)
    negative_mask = tf.less(y_prime, 0)
    zero_mask = tf.equal(y_prime, 0)
    
    sum_ = tf.reduce_sum(tf.boolean_mask(Q_fn(y_prime), positive_mask)) + \
           tf.reduce_sum(1 - tf.boolean_mask(Q_fn(tf.abs(y_prime)), negative_mask)) + \
           0.5 * tf.reduce_sum(tf.cast(zero_mask, tf.float32))
    
    m = tf.cast(tf.size(y), tf.float32)
    return sum_ / m


def huber_loss(x, delta):
    abs_x = tf.abs(x)
    return tf.where(abs_x < delta, 0.5 * tf.square(x), delta * (abs_x - 0.5 * delta))


def huber_loss_derivative(x, delta):
    return tf.where(x > delta, delta / 2, tf.where(x < -delta, -delta / 2, x))


class FairnessLoss:
    def __init__(self, h, tau, delta, male_user_idx, female_user_idx, male_item_idx, female_item_idx, type_='ours'):
        self.h = h
        self.tau = tau
        self.delta = delta
        self.type_ = type_
        self.user = {'M': male_user_idx, 'F': female_user_idx}
        self.item = {'M': male_item_idx, 'F': female_item_idx}

    def DEE(self, y_hat):
        backward_loss = 0
        
        for gender_key in ['M', 'F']:
            for item_key in ['M', 'F']:
                gender_idx = self.user[gender_key]
                item_idx = self.item[item_key]
                
                y_hat_gender = tf.gather(y_hat, gender_idx)
                y_hat_gender_item = tf.gather(y_hat_gender, item_idx, axis=1)

                y_hat_detach = tf.stop_gradient(y_hat)
                y_hat_gender_item_detach = tf.stop_gradient(y_hat_gender_item)
                
                prob_diff_z = normal_cdf(y_hat_detach, self.h, self.tau) - normal_cdf(y_hat_gender_item_detach, self.h, self.tau)
                _dummy = huber_loss_derivative(prob_diff_z, self.delta)
                
                normal_pdf_all = normal_pdf((self.tau - y_hat_detach) / self.h)
                normal_pdf_group = normal_pdf((self.tau - y_hat_gender_item_detach) / self.h)
                
                _dummy *= (tf.tensordot(tf.reshape(normal_pdf_all, [-1]), tf.reshape(y_hat, [-1]), 1) / (self.h * tf.size(y_hat, out_type=tf.float32))) - \
                          (tf.tensordot(tf.reshape(normal_pdf_group, [-1]), tf.reshape(y_hat_gender_item, [-1]), 1) / (self.h * tf.cast(tf.size(y_hat_gender_item), tf.float32)))
                
                backward_loss += _dummy
        return backward_loss

    def __call__(self, y_hat):
        if self.type_ == 'ours':
            return self.DEE(y_hat)