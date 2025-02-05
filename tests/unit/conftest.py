import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def mock_data() -> pd.DataFrame:
    ages = np.array([20, 35, 50] * 4)
    return pd.DataFrame(
        {
            "B_COUNTRY_ALPHA": ["DEU"] * 6 + ["USA"] * 6,
            "Q260": [1] * 3 + [2] * 3 + [1] * 3 + [2] * 3,
            "Q261": 2025 - ages,
            "Q262": ages
        }
    )