import pandas as pd
import pytest
from support_scripts import clean

def test_clean_handles_nulls():
    df = pd.DataFrame({"a": [1, None, 3]})
    cleaned = clean.clean_data(df)
    assert not cleaned.isnull().values.any()
