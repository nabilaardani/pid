"""Titik masuk aplikasi Streamlit.  Jalankan:  streamlit run app.py"""
import streamlit as st

st.set_page_config(
    page_title="PID Iklim Mikro Ruang Arsip",
    page_icon="🗄️",
    layout="wide",
)

from dashboard import render  # noqa: E402  (harus setelah set_page_config)

render()
