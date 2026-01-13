import random
import time

def simulate_live_sensor(df, vitals_cols=None, delay=0.1):
    """
    Simulate real-time streaming of ICU patient data.
    """
    if vitals_cols is None:
        vitals_cols = ['preop_hb', 'preop_cr', 'preop_gluc']
    
    for i in range(len(df)):
        row = df.iloc[i].copy()
        for col in vitals_cols:
            if col in row:
                row[col] += random.uniform(-0.5, 0.5)
        yield row
        time.sleep(delay)
