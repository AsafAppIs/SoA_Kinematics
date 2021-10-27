import numpy as np
import matplotlib.pyplot as plt

# this function gets statistic data regarding total movement 
# and plot error bar of mean, std + minmax

def plot_total_movement(stats):
    
    # give meaningfull names to stats    
    means = stats[0]
    std = stats[1]
    mins = stats[2]
    maxes = stats[3]
    
    
    #plot error bar histogram
    fig, ax = plt.subplots(figsize=(12,6))
    plt.errorbar(np.arange(1, 22), means, std, fmt='ok', lw=3)
    plt.errorbar(np.arange(1, 22), means, [means - mins, maxes - means], fmt='.k', ecolor='gray', lw=1)

    ax.set_title("Total Movement description", fontsize=20, pad=20)
    ax.set_xlabel("Trial Length", fontsize=16, labelpad=20)
    ax.set_ylabel("Total Movement" , fontsize=16, labelpad=20)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.grid(False)
    
    ax.set_xticks(np.arange(1, 22))