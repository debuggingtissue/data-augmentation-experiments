import matplotlib.pyplot as plt
import numpy as np
from fastai.vision.all import *
from utils import path_utils


def get_plot_lr_find(recorder, dataset_name, skip_end=5):
    "Plot the result of an LR Finder test (won't work if you didn't do `learn.lr_find()` before)"
    lrs = recorder.lrs if skip_end == 0 else recorder.lrs[:-skip_end]
    losses = recorder.losses if skip_end == 0 else recorder.losses[:-skip_end]
    fig, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel(f"Learning Rate for {dataset_name}")
    ax.set_xscale('log')
    return fig


def get_plot_loss(recorder, dataset_name, skip_start=0, with_valid=True):
    fig, ax = plt.subplots()
    ax.plot(list(range(skip_start, len(recorder.losses))), recorder.losses[skip_start:], label='train')
    if with_valid:
        idx = (np.array(recorder.iters) < skip_start).sum()
        ax.plot(recorder.iters[idx:], L(recorder.values[idx:]).itemgot(1), label='valid')
        ax.legend()
        ax.set_xlabel(f'Learning Rate for {dataset_name}')
    return fig


def get_batch_plot(dataloader, b=None, max_n=9, ctxs=None, show=True, unique=False, **kwargs):
    if unique:
        old_get_idxs = dataloader.get_idxs
    dataloader.get_idxs = lambda: Inf.zeros
    if b is None: b = dataloader.one_batch()
    if not show:
        return dataloader._pre_show_batch(b, max_n=max_n)
    show_batch(*dataloader._pre_show_batch(b, max_n=max_n), ctxs=ctxs, max_n=max_n, **kwargs)
    if unique: dataloader.get_idxs = old_get_idxs
