import pandas as pd
import numpy as np
import time

def simulate_live_sensor(patient_row, delay=1):
    """
    Simulates live ICU data for ONE patient.
    subjectid NEVER changes.
    Only vitals fluctuate slightly.
    """

    base = patient_row.copy()

    while True:
        live = base.copy()

        # Add small physiological noise (realistic ICU behavior)
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
