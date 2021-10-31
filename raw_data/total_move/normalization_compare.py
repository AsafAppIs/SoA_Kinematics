import raw_data.total_move.total_movement_extraction as tm_extract
import raw_data.total_move.total_movement_stats as tm_stats

def normalization_compare():
    clean_movement = tm_extract.all_subjects_data(list_mode=False, normalize=True)
    movement = tm_extract.all_subjects_data(list_mode=False, normalize=False)
    
    clean_stats = tm_stats.total_movement_stats_calculation(clean_movement)
    regular_stats = tm_stats.total_movement_stats_calculation(movement)

    for i in range(21):
        print(f"point {i+1:2}: before: {regular_stats[1,i]:.3f} after:{clean_stats[1,i]:.3f}")
        print(f"Decrease of {100*(1 - clean_stats[1,i]/regular_stats[1,i]):.3f}%")
    
    return regular_stats, clean_stats