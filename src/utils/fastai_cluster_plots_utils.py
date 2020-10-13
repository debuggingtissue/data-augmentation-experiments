def get_plot_lr_find(recorder, skip_end=5):
    "Plot the result of an LR Finder test (won't work if you didn't do `learn.lr_find()` before)"
    lrs    = recorder.lrs    if skip_end==0 else recorder.lrs   [:-skip_end]
    losses = recorder.losses if skip_end==0 else recorder.losses[:-skip_end]
    fig, ax = plt.subplots(1,1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    return fig

def get_plot_loss(recorder, skip_start=0, with_valid=True):
    fig, ax = plt.subplots()
    ax.plot(list(range(skip_start, len(recorder.losses))), recorder.losses[skip_start:], label='train')
    if with_valid:
        idx = (np.array(recorder.iters)<skip_start).sum()
        ax.plot(recorder.iters[idx:], L(recorder.values[idx:]).itemgot(1), label='valid')
        ax.legend()
    return fig