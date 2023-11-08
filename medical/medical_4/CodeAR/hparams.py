################################
# Experiment Parameters        #
################################
output_directory = './training_log'
iters_per_checkpoint = 10000

################################
# Model Parameters             #
################################
orig_dim = 104
label_dim = 4
fixed_len = 24
hidden_dim = 512
bottleneck_dim = 8

n_codes = 2048
stride = 2
dropout = 0.2

################################
# Optimization Hyperparameters #
################################
epochs = 1000
batch_size = 128
learning_rate = 1e-4
model_type = None