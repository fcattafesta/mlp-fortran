#!/usr/bin/env python3
"""
    This script is used to plot the metrics after the training
"""
import os
import matplotlib.pyplot as plt
import numpy as np

def plotResults(results, save_path='./'):

    if save_path != './':
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    epochs, train_loss, test_loss, train_acc, test_acc = np.loadtxt(results, delimiter=',', unpack=True, skiprows=1)

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(epochs, train_loss, label='Train Loss', color='tomato', lw=2)
    ax.plot(epochs, test_loss, label='Test Loss', color='dodgerblue', lw=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend(frameon=False)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path + 'loss.pdf')

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(epochs, train_acc, label='Train Accuracy', color='tomato', lw=2)
    ax.plot(epochs, test_acc, label='Test Accuracy', color='dodgerblue', lw=2)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend(frameon=False)
    ax.tick_params(axis='both', which='both', direction='in', top=True, right=True)
    ax.grid(True, linestyle='--', alpha=0.5)
    plt.savefig(save_path + 'accuracy.pdf')

if __name__ == "__main__":

    import sys
    if len(sys.argv) != 2:
        raise ValueError("Usage: python plotResults.py <results_file>")
    results = sys.argv[1]
    plotResults(results, save_path='figures/')