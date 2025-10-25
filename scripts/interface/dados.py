# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd

@st.cache_data
def carregar_dados():
    """Carrega o arquivo parquet dos TCCs e valida colunas essenciais."""
    try:
        df = pd.read_parquet("scripts\interface\\tccs_dashboard.parquet")
        required_cols = [
            'titulo', 'autores', 'ano', 'instituicao',  
            'resumo', 'resumo_processado', 'curso', 'nome_topico', 'orientador'
        ]
        missing = [c for c in required_cols if c not in df.columns]
        if missing:
            st.error(f"Colunas faltando no arquivo: {missing}")
            st.stop()

        # Normalizar ano
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
        df = df.dropna(subset=['ano'])
        df['ano'] = df['ano'].astype(int)

        return df

    except FileNotFoundError:
        st.error("Arquivo 'tccs_dashboard.parquet' não encontrado no diretório atual.")
        st.stop()

    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        st.stop()
