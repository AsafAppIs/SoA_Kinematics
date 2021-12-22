import numpy as np
import pytest
import timeseries_data.read_timeseries as rt
import soa_stats.soa_stats as soa



def test_sensitivity():
    results = soa.subject_sensitivity_stats(101)
    
    assert results[0] == pytest.approx(.967, 0.02)
    assert results[1] == pytest.approx(-.484, 0.02)
    
    assert results[2] == pytest.approx(.967, 0.02)
    assert results[3] == pytest.approx(-.484, 0.02)
    
    assert results[4] == pytest.approx(.967, 0.02)
    assert results[5] == pytest.approx(-.484, 0.02)
    
    assert results[6] == pytest.approx(.293, rel=0.09)
    assert results[7] == pytest.approx(-.821, rel=0.09)
    
    assert results[8] == pytest.approx(.967, rel=0.09)
    assert results[9] == pytest.approx(-.484, rel=0.09)

    assert results[10] == pytest.approx(1.642, rel=0.1)
    assert results[11] == pytest.approx(-.146, rel=0.1)


def test_agency():
    data = rt.read_subject(101)
    data = np.array(data)
    results = soa.subject_agency_statistics(data)
    
    assert results[0] == pytest.approx(2/3)
    assert results[1] == pytest.approx(2/3)
    
    assert results[2] == pytest.approx(.967, 0.02)
    assert results[3] == pytest.approx(-.484, 0.02)