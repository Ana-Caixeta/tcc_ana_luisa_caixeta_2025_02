# -*- coding: utf-8 -*-
import streamlit as st

def aplicar_estilo():
    """Insere CSS customizado para o dashboard."""
    st.markdown("""
    <style>
    .stApp { background-color: #F6F6F6; }
    [data-testid="stSidebar"] { background-color: #FFFFFF; }
    .main-header {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        padding: 25px; border-radius: 10px; color: white !important; text-align: center; margin-bottom: 20px;
    }
    .main-header h1 { margin: 0; font-size: 2.2em; color: white !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; background-color: #FFFFFF; padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; background-color: #F6F6F6; border-radius: 8px; color: #1A1A1A; font-weight: 600; }
    .stTabs [aria-selected="true"] { background-color: #4A90E2 !important; color: white !important; }
    [data-testid="stMetricValue"] { font-size: 2em; color: #1A1A1A; }
    [data-testid="stDataFrame"] { background-color: #FFFFFF; }
    </style>
    """, unsafe_allow_html=True)
