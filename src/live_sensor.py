import pandas as pd
import numpy as np
import time

def simulate_live_data(df, delay=2):
    """
    Generator that yields one patient row at a time
    to simulate real-time sensor data.
    """
    for _, row in df.iterrows():
        # simulate slight sensor variation
        row["heart_rate"] += np.random.randint(-3, 4)
        row["oxygen_level"] += np.random.randint(-1, 2)
        row["bp_systolic"] += np.random.randint(-5, 6)

        yield row
        time.sleep(delay)
