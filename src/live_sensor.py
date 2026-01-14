import pandas as pd
import time
import random

def simulate_live_sensor(patient_row):
    base = patient_row.copy()

    while True:
        noisy = base.copy()

        for col in noisy.index:
            if isinstance(noisy[col], (int, float)):
                noisy[col] += random.uniform(-0.5, 0.5)

        yield pd.DataFrame([noisy])
        time.sleep(1)
