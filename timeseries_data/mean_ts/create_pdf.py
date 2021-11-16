import timeseries_data.util.util as util
import timeseries_data.util.comparisons as cmp
from timeseries_data.read_timeseries import read_subject
from timeseries_data.mean_ts.subject_classifier import conditional_mean_ts
from timeseries_data.mean_ts.plot_mean_ts import plot_means, plot_heatmap, plot_mean_with_error
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import timeseries_data.configurations as cfg





if __name__ =="__main__":
    fun = cmp.correlation_ts
    for j, conf in enumerate(cfg.class_configurations):
        for ts in range(11):
            file_name = cfg.class_configurations_names[j] + f" & {cfg.distal_names[ts]}.pdf"
            pp = PdfPages(cfg.pdf_path + file_name)
            
            fig = plt.figure(tight_layout=True)
            
            for i in range(1,26):
                data = read_subject(i)
                data = util.subject_filter_medial(data)
                
                if isinstance(conf, tuple):
                    first_class_mean, first_class_ste, second_class_mean, second_class_ste = conditional_mean_ts(data, idx=conf[1], class_dict=conf[2])
                else:
                    first_class_mean, first_class_ste, second_class_mean, second_class_ste = conditional_mean_ts(data, config=conf)
            
                first_ts = util.get_timeseries(first_class_mean, ts)
                second_ts = util.get_timeseries(second_class_mean, ts)
                
                first_ts_se = util.get_timeseries(first_class_ste, ts)
                second_ts_se = util.get_timeseries(second_class_ste, ts)
                
                ax = fig.add_subplot(5,5,i)
                ax = plot_mean_with_error(first_ts, first_ts_se, second_ts, second_ts_se, title=f"participant {i}", ax=ax, small=True)
                
            pp.savefig(fig)
            
            fig = plt.figure(tight_layout=True)
            
            for i in range(26,42):
                data = read_subject(i)
                data = util.subject_filter_medial(data)
                
                if isinstance(conf, tuple):
                    first_class_mean, first_class_ste, second_class_mean, second_class_ste = conditional_mean_ts(data, idx=conf[1], class_dict=conf[2])
                else:
                    first_class_mean, first_class_ste, second_class_mean, second_class_ste = conditional_mean_ts(data, config=conf)
            
                first_ts = util.get_timeseries(first_class_mean, ts)
                second_ts = util.get_timeseries(second_class_mean, ts)
                
                first_ts_se = util.get_timeseries(first_class_ste, ts)
                second_ts_se = util.get_timeseries(second_class_ste, ts)
                
                ax = fig.add_subplot(5,5,i-25)
                ax = plot_mean_with_error(first_ts, first_ts_se, second_ts, second_ts_se, title=f"participant {i}", ax=ax, small=True)
                
            pp.savefig(fig)
            
            pp.close()        

            

    
    
