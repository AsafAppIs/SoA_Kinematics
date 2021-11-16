import timeseries_data.util.util as util
import timeseries_data.util.comparisons as cmp
from timeseries_data.read_timeseries import read_subject
from timeseries_data.mean_ts.subject_classifier import conditional_mean_ts, all_mean_ts
from timeseries_data.mean_ts.plot_mean_ts import plot_means, plot_heatmap, plot_mean_with_error
from timeseries_data.mean_ts.ts_comparison import timeseries_compare
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt



if __name__ =="__main__":
    participant = 1
    config = "manipulation_t"
    idx=1
    dic = {0:0, 3:1}
    ts = 9
    
    first_class_mean, first_class_ste, second_class_mean, second_class_ste = all_mean_ts(idx=idx, class_dict=dic)
    
    first_ts = util.get_timeseries(first_class_mean, ts)
    second_ts = util.get_timeseries(second_class_mean, ts)
    
    first_ts_se = util.get_timeseries(first_class_ste, ts)
    second_ts_se = util.get_timeseries(second_class_ste, ts)
    
    plot_mean_with_error(first_ts, first_ts_se, second_ts, second_ts_se, small=False)
    
    # matrix = timeseries_compare(data, fun)
        
    #plot_heatmap(matrix, title=f"participant {participant}")