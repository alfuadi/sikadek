import streamlit as st
from streamlit_folium import st_folium
import folium

def select_location():
    m = folium.Map(location=[-2, 117], zoom_start=5)
    m.add_child(folium.LatLngPopup())
    output = st_folium(m, height=500)
    if output and 'last_clicked' in output:
        lat = output['last_clicked']['lat']
        lon = output['last_clicked']['lng']
        return lat, lon
    return None, None
