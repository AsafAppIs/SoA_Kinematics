import pytest
from raw_data.preprocessing import preprocess_subject
@pytest.mark.parametrize("experiment_num, participant_num, trial_num, trial_type, answer, x1, z2",
                         [(1, 7, 36, 4, 0, -0.005185492, 0.3565223), 
                          (1, 7, 122, 3, 1, -0.006196007, 0.3548425), (2, 15, 22, 5, 0, 0.01689812, 0.3290013),
                          (2, 15, 36, 6, 0, 0.01007741, 0.3281931), (2, 15, 122, 3, 0, 0.003826248, 0.3351234)])
def test_preprocessing(experiment_num, participant_num, trial_num, trial_type, answer, x1, z2):
    data = preprocess_subject(experiment_num, participant_num)
    trial = data[data[:,0] == trial_num].squeeze()
    
    assert trial[1] == pytest.approx(trial_type)
    
    assert trial[2] == pytest.approx(answer)
    
    assert trial[3] == pytest.approx(x1)
    
    assert trial[-1] == pytest.approx(z2)