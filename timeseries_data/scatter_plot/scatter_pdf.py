from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import timeseries_data.configurations as cfg
import numpy as np
import pandas as pd



def scatter_trials_by_type(subject_num, ax):
    data = pd.read_csv(cfg.special_feature_path + "participant" + str(subject_num) + ".csv", header=None)
    data_types = [data[data.iloc[:,1] == i] for i in range(7)]
    
    for i in range(7):
        df = data_types[i]
        agency = df[df.iloc[:,2] == 1]
        no_agency = df[df.iloc[:,2] == 0]
        for j, axes in enumerate(ax):
            x = np.random.uniform(i-.15, i+.15, size=len(agency))
            axes.scatter(x, agency.iloc[:, cfg.header_size + j], s=8, color='blue', marker="o")
            x = np.random.uniform(i-.15, i+.15, size=len(no_agency))
            axes.scatter(x, no_agency.iloc[:, cfg.header_size + j], s=8, color='green', marker="x")
    
    ax[0].set_title(f"Y_LOC s{subject_num}", pad=5, fontsize=16)
    ax[1].set_title(f"Z_LOC {subject_num}", pad=5, fontsize=16)
    ax[2].set_title(f"TOTAL_ACC {subject_num}", pad=5, fontsize=16)
    
    for i in range(3):
        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].spines['bottom'].set_color('black')
        ax[i].spines['left'].set_color('black')
        ax[i].spines['bottom'].set_linewidth(0.5)
        ax[i].spines['left'].set_linewidth(0.5)
        ax[i].grid(False)
        ax[i].set_xlabel("Manipulation type", fontsize=14)
        ax[i].tick_params(axis = 'y', labelsize=8)
        ax[i].set_xticklabels(["","no manipulation", "100ms", "200ms", "300ms", "6deg", "10deg", "14deg"], fontsize=8, rotation=45)
    
    



if __name__ =="__main__":
    file_name = "participants.pdf"
    pp = PdfPages(cfg.scatter_path + file_name)
    fig = plt.figure(tight_layout=True, figsize=(30, 175))
    

    for i in range(cfg.num_of_participants):
        ax1 = fig.add_subplot(41,3,1 + (i*3))
        ax2 = fig.add_subplot(41,3,2 + (i*3))
        ax3 = fig.add_subplot(41,3,3 + (i*3))
    
        scatter_trials_by_type(i+1, np.array([ax1,ax2,ax3]))
    
    fig.subplots_adjust(left=0.125,
            bottom=0.2, 
            right=0.8, 
            top=0.8, 
            wspace=0.2, 
            hspace=14)

    pp.savefig(fig)
    pp.close()       