import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeseries_data.configurations as cfg

def correlation_heatmap(corr, title="Correlation Heatmap"):
    # create mask
    mask = np.triu(np.ones_like(corr, dtype=np.bool))

    # adjust mask and df
    mask = mask[1:, :-1]
    corr = corr[1:,:-1]

    fig, ax = plt.subplots(figsize=(22,20))
    
    sns.heatmap(corr, mask=mask, cmap="Reds", annot=True, fmt=".2f",
                vmin=-1, vmax=1, square=True, linewidth=2, cbar_kws={"shrink": .8})
    ax.set_title(title, fontsize=20, pad=20)
    
    plt.xticks(np.arange(len(cfg.full_names)-1) + .5, labels=cfg.full_names[:-1], rotation=45)
    plt.yticks(np.arange(len(cfg.full_names)-1) + .5, labels=cfg.full_names[1:], rotation=0)
    #plt.xticks(rotation=70)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.grid(False)