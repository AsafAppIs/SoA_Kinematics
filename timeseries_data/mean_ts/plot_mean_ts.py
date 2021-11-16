import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeseries_data.configurations as cfg


def plot_mean_with_error(first_ts_mean, first_ts_ste, second_ts_mean, second_ts_ste, title="", ax=False, small=False):
    font_title = 6 if small else 16
    font_label = 4 if small else 12
    font_def = 4 if small else 11
    font_ticks = 3 if small else 10
    font_legend = 4 if small else 10
    line_width = 0.2 if small else 1
    
    plt.rc('font', size=font_def)          # controls default text sizes
    plt.rc('axes', titlesize=font_title)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_label)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_ticks)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_ticks)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_legend)    # legend fontsize

    
    if not ax:
        fig, ax = plt.subplots(figsize=(12,6))
    
    # calculate y axis limits
    max_std = max(max(first_ts_ste), max(second_ts_ste))
    y_min = min(min(first_ts_mean), min(second_ts_mean))
    y_min  -= max_std*3
    
    y_max = max(max(first_ts_mean), max(second_ts_mean))
    y_max  += max_std*3
    
    # x 
    x = np.arange(0, cfg.ts_length)
    
    # plot first
    ax.plot(x, first_ts_mean, color='r', linewidth=line_width)
    ax.fill_between(x, first_ts_mean + first_ts_ste, first_ts_mean - first_ts_ste, color='r', alpha=.3)
    
    # plot second
    ax.plot(x, second_ts_mean, color='b', linewidth=line_width)
    ax.fill_between(x, second_ts_mean + second_ts_ste, second_ts_mean - second_ts_ste, color='b', alpha=.3)
    
    
    ax.set_title(title, fontsize=font_title)
    ax.set_xlabel("time", fontsize=font_label, labelpad=font_label/2)
    ax.set_ylabel("magnitude", fontsize=font_label, labelpad=font_label/2)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.set_ylim(y_min, y_max)
    ax.grid(False)
    
    
    return ax



def plot_means(first_ts, second_ts, title=""):
    fig, ax = plt.subplots(figsize=(12,6), ncols=2)
    
    # calculate y axis limits
    y_min = min(min(first_ts), min(second_ts))
    y_min  -= abs(y_min)*.1
    
    y_max = max(max(first_ts), max(second_ts))
    y_max  += abs(y_max)*.1
    
    fig.suptitle(title, fontsize=16)
    
    ax[0].plot(first_ts)
    ax[0].set_title("First timeseries", fontsize=14, pad=20)
    ax[0].set_xlabel("time", fontsize=11)
    ax[0].set_ylabel("magnitude", fontsize=11)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_color('black')
    ax[0].spines['left'].set_color('black')
    ax[0].spines['bottom'].set_linewidth(0.5)
    ax[0].spines['left'].set_linewidth(0.5)
    ax[0].set_ylim(y_min, y_max)
    ax[0].grid(False)
    
    ax[1].plot(second_ts)
    ax[1].set_title("Second timeseries", fontsize=14, pad=20)
    ax[1].set_xlabel("time", fontsize=11)
    ax[1].set_ylabel("magnitude", fontsize=11)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_color('black')
    ax[1].spines['left'].set_color('black')
    ax[1].spines['bottom'].set_linewidth(0.5)
    ax[1].spines['left'].set_linewidth(0.5)
    ax[1].set_ylim(y_min, y_max)
    ax[1].grid(False)
    
    
    
def plot_heatmap(matrix, title="", full=False):
    x_labels = cfg.full_names if full else cfg.distal_names
    y_label = cfg.class_configurations_names
    
    fig, ax = plt.subplots(figsize=(22,20))
    
    
    sns.heatmap(matrix, cmap="Reds", annot=True, fmt=".2f",
            vmin=-1, vmax=1, square=True, linewidth=2, cbar_kws={"shrink": .8})
    ax.set_title(title, fontsize=20, pad=20)
    
    plt.xticks(np.arange(len(x_labels)) + .5, labels=x_labels, rotation=45)
    plt.yticks(np.arange(len(y_label)) + .5, labels=y_label, rotation=0)
    #plt.xticks(rotation=70)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.grid(False)
