import pandas as pd
import numpy as np
from datetime import datetime, timedelta, UTC
from scipy.spatial.distance import cdist
from siphon.simplewebservice.wyoming import WyomingUpperAir
from metpy.units import units
from metpy.calc import (
    relative_humidity_from_dewpoint, wind_speed, wind_direction, lcl, lfc, parcel_profile, surface_based_cape_cin,
    k_index, lifted_index, showalter_index, ccl, sweat_index, vertical_totals
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import requests
from bs4 import BeautifulSoup

# ------------------------------
# 1. SET KOORDINAT MANUAL
# ------------------------------
target_lat = -6.2
target_lon = 106.8

# ------------------------------
# 2. CARI STASIUN TERDEKAT
# ------------------------------
def find_nearest_station(lat, lon, station_file):
    df = pd.read_csv(station_file)
    coords = df[['lat', 'lon']].values
    target = np.array([[lat, lon]])
    dists = cdist(coords, target, metric='euclidean')
    nearest_index = np.argmin(dists)
    return df.iloc[nearest_index]

sounding_station = find_nearest_station(target_lat, target_lon, 'SoundingStations.csv')
synop_station = find_nearest_station(target_lat, target_lon, 'SynopStations.csv')

print(f"Stasiun sounding terdekat: {sounding_station['loc']} ({sounding_station['id']})")
print(f"Stasiun hujan terdekat: {synop_station['loc']} ({synop_station['id']})")

# ------------------------------
# 3. AMBIL DATA SOUNDING 10 HARI
# ------------------------------
def fetch_sounding_data(wmo_id, days=30):
    results = []
    for i in range(days + 1):
        hour=0
        try:
            date = datetime.now(UTC) - timedelta(days=i)
            time = datetime(date.year, date.month, date.day, hour)
            df = WyomingUpperAir.request_data(time, str(wmo_id))
            df = df.dropna(subset=["pressure", "temperature", "dewpoint", "height"])
            df = df.drop_duplicates(subset=['pressure'])
            p = df["pressure"].values * units.hPa
            T = df["temperature"].values * units.degC
            Td = df["dewpoint"].values * units.degC
            u = df["u_wind"].values * units.knots
            v = df["v_wind"].values * units.knots
            df["RH"] = relative_humidity_from_dewpoint(T,Td).to('percent')

            sort_idx = np.argsort(p.magnitude)[::-1]
            p, T, Td, u, v = p[sort_idx], T[sort_idx], Td[sort_idx], u[sort_idx], v[sort_idx]

            p_sfc, T_sfc, Td_sfc = p[0], T[0], Td[0]
            lcl_p, lcl_T = lcl(p_sfc, T_sfc, Td_sfc)
            prof = parcel_profile(p, T_sfc, Td_sfc)
            cape, cin = surface_based_cape_cin(p, T, Td)
            ki = k_index(p, T, Td)
            li = lifted_index(p, T, Td)
            si = showalter_index(p, T, Td)
            def extract(var, level):
                val = df[df["pressure"] == level][var].values
                return val[0] if len(val) > 0 else np.nan
            rh850 = extract("RH", 850)
            rh700 = extract("RH", 700)
            rh500 = extract("RH", 500)
            wspd850 = np.sqrt(extract("u_wind", 850)**2 + extract("v_wind", 850)**2)
            wspd700 = np.sqrt(extract("u_wind", 700)**2 + extract("v_wind", 700)**2)
            wspd500 = np.sqrt(extract("u_wind", 500)**2 + extract("v_wind", 500)**2)
            ccl_ = ccl(p, T, Td)
            lcl_ = lcl(p, T, Td)
            lfc_ = lfc(p, T, Td)
            sweat = sweat_index(p, T, Td, wind_speed(u,v), wind_direction(u,v))
            vert = vertical_totals(p, Td)

            results.append({
                "time": time.date(),
                "jam": time.hour,
                "cape": cape.magnitude,
                "cin": cin.magnitude,
                "ki": ki.magnitude,
                "li": li.magnitude[0],
                "si": si.magnitude[0],
                "rh850": rh850,
                "rh700": rh700,
                "rh500": rh500,
                "wspd850": wspd850,
                "wspd700": wspd700,
                "wspd500": wspd500,
                "ccl_":ccl_[0].magnitude,
                "lcl_":(lcl_[0].magnitude)[0],
                "lfc_":lfc_[0].magnitude,
                "sweat":sweat.magnitude[0],
                "vert":vert.magnitude
            })
        except Exception as e:
            print(f"  ⛔ {time} gagal: {e}")
            continue
    return pd.DataFrame(results)

sounding_df = fetch_sounding_data(sounding_station['id'])
##sounding_df.to_csv('D:/Project/Konten/WebPrediksiHujan_Rason/sounding.csv')
##sounding_df = pd.read_csv('D:/Project/Konten/WebPrediksiHujan_Rason/sounding.csv')

# ------------------------------
# 4. SCRAPING CURAH HUJAN OGIMET
# ------------------------------
def fetch_rain_data(wmo_id, periods=30):
    validdate = datetime.now(UTC)
    year, month, day, hour = validdate.year, validdate.month, validdate.day, validdate.hour
    url = f"https://www.ogimet.com/cgi-bin/gsynres?lang=en&ind={wmo_id}&decoded=yes&ndays={periods}&ano={year}&mes={month}&day={day}&hora={hour}"
    tables = pd.read_html(url, header=[0])
    df_raw = tables[2].copy()  # Tabel ketiga biasanya data utama

    df_raw.columns = [' '.join([str(c1).strip()]).strip() for c1 in df_raw.columns]

    date_col = [col for col in df_raw.columns if 'Date' in col][0]
    hour_col = [col for col in df_raw.columns if 'Hour' in col or 'Time' in col or 'Date.1' in col][0]
    prec_col = [col for col in df_raw.columns if 'Prec' in col][0]

    df_weather = df_raw[[date_col, hour_col, prec_col]].copy()
    df_weather.columns = ['Date', 'Hour', 'Prec_mm']

    df_weather['time'] = pd.to_datetime(df_weather['Date'] + ' ' + df_weather['Hour'], errors='coerce')

    def extract_precip(prec_string):
        if pd.isna(prec_string):
            return None
        parts = str(prec_string).split()
        for part in parts:
            if '/3h' in part or '/6h' in part:
                try:
                    if 'Tr' in part:
                        return 0.1
                    else:
                        return float(part.replace('/3h', '').replace('/6h', ''))
                except:
                    return np.nan
            else:
                try:
                    if 'Tr' in part:
                        return 0.1
                    else:
                        return float(part.replace('/24h', ''))
                except:
                    return np.nan

        return 0

    df_weather['prec_3h'] = df_weather['Prec_mm'].apply(extract_precip)
    df_weather = df_weather.sort_values(by='time', ascending=False).reset_index(drop=True)
    df_weather['prec_12h'] = np.nan

    for i, row in df_weather.iterrows():
        ts = row['time']
        if ts.hour in [0, 12]:
            end_time = ts
            start_time = ts - pd.Timedelta(hours=11, minutes=59)
            mask = (df_weather['time'] <= end_time) & (df_weather['time'] > start_time)
            df_weather.loc[i, 'prec_12h'] = df_weather.loc[mask, 'prec_3h'].sum()

    df_weather['ww'] = df_weather['prec_12h'].apply(lambda x: 1 if x > 0 else 0)
    df_weather = df_weather.groupby('Date').max().reset_index()
    df_weather = df_weather.drop(columns=['Hour', 'Prec_mm', 'prec_3h', 'prec_12h'])
    return df_weather

rain_df = fetch_rain_data(synop_station['id'])
##rain_df.to_csv('D:/Project/Konten/WebPrediksiHujan_Rason/rain_df.csv')
##rain_df = pd.read_csv('D:/Project/Konten/WebPrediksiHujan_Rason/rain_df.csv')

# ------------------------------
# 5. GABUNGKAN & LATIH MODEL
# ------------------------------
sounding_df = sounding_df[sounding_df['jam']==0]
sounding_df['Date'] = pd.to_datetime(sounding_df['time']).dt.date
sounding_df = sounding_df.drop(columns=['time','jam'])
rain_df = rain_df.drop(columns=['time'])
rain_df['Date'] = pd.to_datetime(rain_df['Date']).dt.date
training_df = pd.merge(sounding_df, rain_df, on=['Date'], how='inner')
training_df = training_df.dropna(how='any')

if training_df.empty:
    print("Data tidak cukup untuk pelatihan.")
    exit()

X = training_df.drop(columns=['Date', 'ww'])
y = training_df['ww']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(random_state=0).fit(X_scaled, y)
mlp = MLPClassifier(hidden_layer_sizes=(100,100),max_iter=1000, random_state=0, early_stopping=False).fit(X_scaled, y)

auc_rf = roc_auc_score(y, rf.predict_proba(X_scaled)[:,1])
auc_mlp = roc_auc_score(y, mlp.predict_proba(X_scaled)[:,1])

print(f"ROC-AUC RF: {auc_rf:.2f}, MLP: {auc_mlp:.2f}")

# ------------------------------
# 6. PREDIKSI SOUNDING TERAKHIR
# ------------------------------
latest_data = sounding_df.sort_values(by=["Date"]).iloc[-1:]
X_latest = latest_data.drop(columns=["Date"])
X_latest_scaled = scaler.transform(X_latest)

rf_prob = rf.predict_proba(X_latest_scaled)[0][1] * 100
mlp_prob = mlp.predict_proba(X_latest_scaled)[0][1] * 100

print(f"\nPrediksi peluang hujan 24 jam ke depan ({}-{}):")
print(f"Random Forest : {rf_prob:.1f}%")
print(f"MLP           : {mlp_prob:.1f}%")


