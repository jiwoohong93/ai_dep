import os
import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def get_writer(output_directory, log_directory):
    if not os.path.exists(output_directory):
        os.mkdir(output_directory)
        
    logging_path=f'{output_directory}/{log_directory}'
    if not os.path.exists(logging_path):
        os.mkdir(logging_path)
        
    writer = SummaryWriter(logging_path)
    return writer


def plot_image(preds, targets, generated=None):
    fig = plt.figure(figsize=(12,4))
    plt.plot(preds, label='preds')
    plt.plot(targets, label='targets')
    if generated is not None:
        plt.plot(generated, label='generated')
    plt.legend()
    return fig


def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print(f"Saving model and optimizer state at iteration {iteration} to {filepath}")
    torch.save({'iteration': iteration,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}" .format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration