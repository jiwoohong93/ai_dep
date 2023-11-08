################################
# Experiment Parameters        #
################################
output_directory = './training_log'
iters_per_checkpoint = 10000

################################
# Model Parameters             #
################################
orig_dim = 3
label_dim=5
fixed_len = 16
hidden_dim = 256

################################
# Optimization Hyperparameters #
################################
epochs = 1000
batch_size = 512
learning_rate = 1e-4