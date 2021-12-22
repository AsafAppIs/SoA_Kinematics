from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import timeseries_data.configurations as cfg
import numpy as np
from timeseries_data.mean_ts.subject_classifier import subject_classifier
from timeseries_data.mean_ts.trial_classifier import trial_classfier_creator
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from kinematic_data.kinematic_to_ts.interpolate import interpolate
import timeseries_data.util.soa_equal as equal

import pylab


def mean_roc(config=False, idx=0, class_dict=False, title='', return_fig=False, ax=False):
    tp1 = []
    tp2 = []
    tp3 = []
    tp4 = []
    tp5 = []
    fp1 = []
    fp2 = []
    fp3 = []
    fp4 = []
    fp5 = []
    auc1 = []
    auc2 = []
    auc3 = []
    auc4 = []
    auc5 = []
    # define classifier function
    classifier = trial_classfier_creator(config=config, idx=idx, class_dict=class_dict)
    
    for i in range(cfg.num_of_participants):
        df = pd.read_csv(cfg.special_feature_path + "participant" + str(1+i) + ".csv", header=None)
        data = np.array(df)
        index1, index2 = subject_classifier(data, classifier)
        # equalize number of trials in each group if the soa condition was choose
        if config == "soa":
            index1, index2 = equal.soa_equalizer_down(data, index1, index2)
        if (len(index1) == 0) or (len(index2) == 0):
            continue
        
        # first measure
        new_data = np.array([(data[i, 3], 1) if (i in index1) else (data[i, 3], 0) if (i in index2) else -1 
                         for i in range(len(data))], dtype=np.object)
        new_data = np.array([x for x in new_data if x is not -1], dtype=np.float32)
        weight1 = len(index1) / len(new_data)
        weight2 = len(index2) / len(new_data)       
        index1_array = new_data[:, 1]
        weight_array = np.array([weight1 if index1_array[i] else weight2 for i in range(len(new_data))])   
        measure_data = new_data[:, 0]        
        fp, tp, _ = roc_curve(index1_array, measure_data, sample_weight=weight_array)  
        auc = roc_auc_score(index1_array, measure_data, sample_weight=weight_array)   
        fp1.append(fp)
        tp1.append(tp)
        auc1.append(auc)
        
        # second measure
        new_data = np.array([(data[i, 4], 1) if (i in index1) else (data[i, 4], 0) if (i in index2) else -1 
                         for i in range(len(data))], dtype=np.object)
        new_data = np.array([x for x in new_data if x is not -1], dtype=np.float32)
        measure_data = new_data[:, 0]        
        fp, tp, _ = roc_curve(index1_array, measure_data, sample_weight=weight_array) 
        auc = roc_auc_score(index1_array, measure_data, sample_weight=weight_array)  
        fp2.append(fp)
        tp2.append(tp)
        auc2.append(auc)
        
        # third measure
        new_data = np.array([(data[i, 5], 1) if (i in index1) else (data[i, 5], 0) if (i in index2) else -1 
                         for i in range(len(data))], dtype=np.object)
        new_data = np.array([x for x in new_data if x is not -1], dtype=np.float32)
        measure_data = new_data[:, 0]        
        fp, tp, _ = roc_curve(index1_array, measure_data, sample_weight=weight_array)  
        auc = roc_auc_score(index1_array, measure_data, sample_weight=weight_array)  
        fp3.append(fp)
        tp3.append(tp)
        auc3.append(auc)
        
        
         # fourth measure
        new_data = np.array([(data[i, 6], 1) if (i in index1) else (data[i, 6], 0) if (i in index2) else -1 
                         for i in range(len(data))], dtype=np.object)
        new_data = np.array([x for x in new_data if x is not -1], dtype=np.float32)
        measure_data = new_data[:, 0]*-1      
        fp, tp, _ = roc_curve(index1_array, measure_data, sample_weight=weight_array)  
        auc = roc_auc_score(index1_array, measure_data, sample_weight=weight_array)  
        fp4.append(fp)
        tp4.append(tp)
        auc4.append(auc)
        
        
        
         # fifth measure
        new_data = np.array([(data[i, 7], 1) if (i in index1) else (data[i, 7], 0) if (i in index2) else -1 
                         for i in range(len(data))], dtype=np.object)
        new_data = np.array([x for x in new_data if x is not -1], dtype=np.float32)
        measure_data = new_data[:, 0]*-1   
        fp, tp, _ = roc_curve(index1_array, measure_data, sample_weight=weight_array)  
        auc = roc_auc_score(index1_array, measure_data, sample_weight=weight_array)  
        fp5.append(fp)
        tp5.append(tp)
        auc5.append(auc)
        
    
    fp1 = np.array([interpolate(x, 60,'linear') for x in fp1])
    fp2 = np.array([interpolate(x, 60,'linear') for x in fp2])
    fp3 = np.array([interpolate(x, 60,'linear') for x in fp3])
    fp4 = np.array([interpolate(x, 60,'linear') for x in fp4])
    fp5 = np.array([interpolate(x, 60,'linear') for x in fp5])
    tp1 = np.array([interpolate(x, 60,'linear') for x in tp1])
    tp2 = np.array([interpolate(x, 60,'linear') for x in tp2])
    tp3 = np.array([interpolate(x, 60,'linear') for x in tp3])
    tp4 = np.array([interpolate(x, 60,'linear') for x in tp4])
    tp5 = np.array([interpolate(x, 60,'linear') for x in tp5])
    
    fp1_mean = np.mean(fp1, axis=0)
    fp2_mean = np.mean(fp2, axis=0)
    fp3_mean = np.mean(fp3, axis=0)
    fp4_mean = np.mean(fp4, axis=0)
    fp5_mean = np.mean(fp5, axis=0)
    tp1_mean = np.mean(tp1, axis=0)
    tp2_mean = np.mean(tp2, axis=0)
    tp3_mean = np.mean(tp3, axis=0)
    tp4_mean = np.mean(tp4, axis=0)
    tp5_mean = np.mean(tp5, axis=0)
    
    auc1_mean = np.mean(auc1)
    auc2_mean = np.mean(auc2)
    auc3_mean = np.mean(auc3)
    auc4_mean = np.mean(auc4)
    auc5_mean = np.mean(auc5)
    
    font_title = 4
    font_label = 4 
    font_def = 4 
    font_ticks = 3 
    font_legend = 4 
    line_width = 0.2 
    
    plt.rc('font', size=font_def)          # controls default text sizes
    plt.rc('axes', titlesize=font_title)     # fontsize of the axes title
    plt.rc('axes', labelsize=font_label)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=font_ticks)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=font_ticks)    # fontsize of the tick labels
    plt.rc('legend', fontsize=font_legend)    # legend fontsize
    
    
    cm = pylab.get_cmap('gist_rainbow')
    
    for i in range(len(fp1)):
        ax[0].plot(fp1[i],tp1[i], alpha=.4, linewidth=.6, color = cm(1.*i/len(fp1)))
    ax[0].plot(fp1_mean, tp1_mean, label=f"MEAN AUC={auc1_mean:.2f}", color="red")
    ax[0].plot([0,1], [0,1], label="Guess line", color="blue", linewidth=1)
    ax[0].set_title(f"Y_LOC {title}", pad=5)
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].spines['bottom'].set_color('black')
    ax[0].spines['left'].set_color('black')
    ax[0].spines['bottom'].set_linewidth(0.5)
    ax[0].spines['left'].set_linewidth(0.5)
    ax[0].grid(False)
    ax[0].legend()
    
    for i in range(len(fp2)):
        ax[1].plot(fp2[i],tp2[i], alpha=.4, linewidth=.6, color = cm(1.*i/len(fp1)))
    ax[1].plot(fp2_mean, tp2_mean, label=f"MEAN AUC={auc2_mean:.2f}", color="red")
    ax[1].plot([0,1], [0,1], label="Guess line", color="blue", linewidth=1)
    ax[1].set_title(f"Z_LOC {title}", pad=5)
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].spines['bottom'].set_color('black')
    ax[1].spines['left'].set_color('black')
    ax[1].spines['bottom'].set_linewidth(0.5)
    ax[1].spines['left'].set_linewidth(0.5)
    ax[1].grid(False)
    ax[1].legend()
    
    for i in range(len(fp3)):
        ax[2].plot(fp3[i],tp3[i], alpha=.4, linewidth=.6, color = cm(1.*i/len(fp1)))
    ax[2].plot(fp3_mean, tp3_mean, label=f"MEAN AUC={auc3_mean:.2f}", color="red")
    ax[2].plot([0,1], [0,1], label="Guess line", color="blue", linewidth=1)
    ax[2].set_title(f"TOTAL_ACC {title}", pad=5)
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)
    ax[2].spines['bottom'].set_color('black')
    ax[2].spines['left'].set_color('black')
    ax[2].spines['bottom'].set_linewidth(0.5)
    ax[2].spines['left'].set_linewidth(0.5)
    ax[2].grid(False)
    ax[2].legend()
    
    
    for i in range(len(fp4)):
        ax[3].plot(fp4[i],tp4[i], alpha=.4, linewidth=.6, color = cm(1.*i/len(fp1)))
    ax[3].plot(fp4_mean, tp4_mean, label=f"MEAN AUC={auc4_mean:.2f}", color="red")
    ax[3].plot([0,1], [0,1], label="Guess line", color="blue", linewidth=1)
    ax[3].set_title(f"Y_LOC movement {title}", pad=5)
    ax[3].spines['top'].set_visible(False)
    ax[3].spines['right'].set_visible(False)
    ax[3].spines['bottom'].set_color('black')
    ax[3].spines['left'].set_color('black')
    ax[3].spines['bottom'].set_linewidth(0.5)
    ax[3].spines['left'].set_linewidth(0.5)
    ax[3].grid(False)
    ax[3].legend()
    
    
    
    for i in range(len(fp5)):
        ax[4].plot(fp5[i],tp5[i], alpha=.4, linewidth=.6, color = cm(1.*i/len(fp1)))
    ax[4].plot(fp5_mean, tp5_mean, label=f"MEAN AUC={auc5_mean:.2f}", color="red")
    ax[4].plot([0,1], [0,1], label="Guess line", color="blue", linewidth=1)
    ax[4].set_title(f"Z_LOC movement {title}", pad=5)
    ax[4].spines['top'].set_visible(False)
    ax[4].spines['right'].set_visible(False)
    ax[4].spines['bottom'].set_color('black')
    ax[4].spines['left'].set_color('black')
    ax[4].spines['bottom'].set_linewidth(0.5)
    ax[4].spines['left'].set_linewidth(0.5)
    ax[4].grid(False)
    ax[4].legend()


if __name__ =="__main__":
    file_name = "roc_curves1.pdf"
    num = 19

    pp = PdfPages(cfg.auc_path + file_name)
    fig = plt.figure(tight_layout=True, figsize=(8, num*2))
    
    for j, conf in enumerate(cfg.class_configurations[:num]):
        ax1 = fig.add_subplot(num,5,1 + (j*5))
        ax2 = fig.add_subplot(num,5,2 + (j*5))
        ax3 = fig.add_subplot(num,5,3 + (j*5))
        ax4 = fig.add_subplot(num,5,4 + (j*5))
        ax5 = fig.add_subplot(num,5,5 + (j*5))
        if isinstance(conf, tuple):
            mean_roc(idx=conf[1], class_dict=conf[2], title=cfg.class_configurations_names[j], return_fig=True, ax=[ax1,ax2,ax3, ax4, ax5])
        else:
            mean_roc(config=conf, title=cfg.class_configurations_names[j], return_fig=True, ax=[ax1,ax2,ax3, ax4, ax5])
    
    fig.subplots_adjust(left=0.125,
            bottom=0.1, 
            right=0.9, 
            top=0.9, 
            wspace=0.2, 
            hspace=2)
    
    pp.savefig(fig)
    '''
    fig = plt.figure(tight_layout=True)
    for j, conf in enumerate(cfg.class_configurations[5:10]):
        ax1 = fig.add_subplot(5,3,1 + (j*3))
        ax2 = fig.add_subplot(5,3,2 + (j*3))
        ax3 = fig.add_subplot(5,3,3 + (j*3))
        if isinstance(conf, tuple):
            mean_roc(idx=conf[1], class_dict=conf[2], title=cfg.class_configurations_names[j+5], return_fig=True, ax=[ax1,ax2,ax3])
        else:
            mean_roc(config=conf, title=cfg.class_configurations_names[j+5], return_fig=True, ax=[ax1,ax2,ax3])
    pp.savefig(fig)
    
    fig = plt.figure(tight_layout=True)
    for j, conf in enumerate(cfg.class_configurations[10:15]):
        ax1 = fig.add_subplot(5,3,1 + (j*3))
        ax2 = fig.add_subplot(5,3,2 + (j*3))
        ax3 = fig.add_subplot(5,3,3 + (j*3))
        if isinstance(conf, tuple):
            mean_roc(idx=conf[1], class_dict=conf[2], title=cfg.class_configurations_names[j+10], return_fig=True, ax=[ax1,ax2,ax3])
        else:
            mean_roc(config=conf, title=cfg.class_configurations_names[j+10], return_fig=True, ax=[ax1,ax2,ax3])
    pp.savefig(fig)


    fig = plt.figure(tight_layout=True)
    for j, conf in enumerate(cfg.class_configurations[15:]):
        ax1 = fig.add_subplot(5,3,1 + (j*3))
        ax2 = fig.add_subplot(5,3,2 + (j*3))
        ax3 = fig.add_subplot(5,3,3 + (j*3))
        if isinstance(conf, tuple):
            mean_roc(idx=conf[1], class_dict=conf[2], title=cfg.class_configurations_names[j+15], return_fig=True, ax=[ax1,ax2,ax3])
        else:
            mean_roc(config=conf, title=cfg.class_configurations_names[j+15], return_fig=True, ax=[ax1,ax2,ax3])
    pp.savefig(fig)
    
    '''
    
    pp.close()        
