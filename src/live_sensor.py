import pandas as pd
import numpy as np
import time

def simulate_live_sensor(patient_row, delay=1):
    """
    Simulate live vitals for ONE patient.
    """
    base = patient_row.copy()
    while True:
        live = base.copy()
        noise_map = {
            "preop_hb": 0.2,
            "preop_gluc": 2,
            "preop_cr": 0.05,
            "intraop_ebl": 10,
            "intraop_uo": 10,
        }
        for col, noise in noise_map.items():
            if col in live and pd.notna(live[col]):
                live[col] = max(0, live[col] + np.random.uniform(-noise, noise))
        yield pd.DataFrame([live])
        time.sleep(delay)
