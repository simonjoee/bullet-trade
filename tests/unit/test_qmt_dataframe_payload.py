import json
from datetime import datetime

import pandas as pd
import pytest

from bullet_trade.server.adapters.qmt import dataframe_to_payload


@pytest.mark.unit
def test_dataframe_to_payload_handles_datetime():
    df = pd.DataFrame(
        {
            "start_date": [pd.Timestamp("2025-01-01")],
            "end_date": [pd.NaT],
            "value": [1],
        }
    )
    payload = dataframe_to_payload(df)
    encoded = json.dumps(payload)
    assert "2025-01-01" in encoded
