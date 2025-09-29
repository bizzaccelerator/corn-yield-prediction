import pandas as pd

from scripts.support_scripts import drift_monitoring


def test_no_drift():
    df1 = pd.DataFrame({"x": [1, 2, 3]})
    df2 = df1.copy()
    drift = drift_monitoring.check_drift(df1, df2)
    assert drift is False


def test_detect_drift():
    df1 = pd.DataFrame({"x": [1, 2, 3]})
    df2 = pd.DataFrame({"x": [100, 200, 300]})
    drift = drift_monitoring.check_drift(df1, df2)
    assert drift is True
