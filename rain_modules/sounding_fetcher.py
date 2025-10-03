from siphon.simplewebservice.wyoming import WyomingUpperAir
from datetime import datetime, timedelta
import pandas as pd

def get_nearest_station(lat, lon):
    # Hardcoded atau pakai file WMO stasiun untuk mencari terdekat
    return '96749'  # contoh: Jakarta

def fetch_sounding_data(station, days=10):
    today = datetime.utcnow()
    times = [(today - timedelta(days=i)).replace(hour=h) 
             for i in range(1, days+1) for h in [0, 12]]
    all_data = []
    for t in times:
        try:
            df = WyomingUpperAir.request_data(t, station)
            # Extract needed parameters
            # Append dict or series ke list
        except Exception:
            continue
    return pd.DataFrame(all_data)
