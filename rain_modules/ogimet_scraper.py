import requests
from bs4 import BeautifulSoup

def get_nearest_ogimet_station(lat, lon):
    # Gunakan WMO stasiun dari Ogimet atau metadata yang Anda miliki
    return '96749'  # contoh

def fetch_rain_data(wmo_id, days=10):
    # Scraping tabel curah hujan dari Ogimet selama 10 hari terakhir
    # Parsing HTML ke dataframe
    return rain_df
