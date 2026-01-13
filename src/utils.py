import pandas as pd
import random
import time

def simulate_live_sensor(df, vitals_cols=['preop_hb', 'preop_cr', 'preop_gluc']):
    """
    Simulate live streaming of ICU data
    """
    for i in range(len(df)):
        row = df.iloc[i].copy()
        # add small random noise
        for col in vitals_cols:
            row[col] += random.uniform(-0.5, 0.5)
        yield row
        time.sleep(0.1)  # simulate real-time delay
