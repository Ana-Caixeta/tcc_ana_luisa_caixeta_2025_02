# -*- coding: utf-8 -*-
import re
from collections import Counter
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LinearRegression

def metric_bold(label, value):
    """Cria métrica com texto e valor em negrito dentro de um card de altura fixa."""
    st.markdown(f"""
    <div style='
        text-align: center;
        background-color: #4A90E2;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
        margin-bottom: 10px;
        height: 120px;
        max-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    '>
        <p title="{label}" style='
            font-size: 1.1em;
            font-weight: bold;
            color: #FFFFFF;
            margin-bottom: 5px;
            hyphens: auto;
        '>
            {label}
        </p>
        <p title="{value}" style='
            font-size: clamp(1.2em, 2em, 2em);
            font-weight: 800;
            color: #FFFFFF;
            margin: 0;
            line-height: 1.2;
            hyphens: auto;
        '>
            {value}
        </p>
    </div>
    """, unsafe_allow_html=True)

def filtrar_dados(df, instituicoes, anos, topicos, cursos):
    """Aplica os filtros selecionados no dataframe."""
    df_f = df.copy()
    if instituicoes:
        df_f = df_f[df_f['instituicao'].isin(instituicoes)]
    if anos:
        df_f = df_f[df_f['ano'].between(anos[0], anos[1])]
    if topicos:
        df_f = df_f[df_f['nome_topico'].isin(topicos)]
    if cursos:
        df_f = df_f[df_f['curso'].isin(cursos)]
    return df_f

def simplificar_topico(nome_topico):
    """Remove prefixos tipo 'Tópico X: ' para exibição curta."""
    return re.sub(r'Tópico \d+: ', '', str(nome_topico))

def extract_keywords(texts, top_n=15):
    """Extrai palavras mais frequentes de uma lista de textos."""
    all_words = []
    for text in texts:
        if isinstance(text, str):
            all_words.extend(text.split())
    word_freq = Counter(all_words)
    return word_freq.most_common(top_n)

def calcular_similaridade(df, idx_referencia, top_n=5):
    """Calcula TCCs similares usando TF-IDF e similaridade cosseno."""
    if len(df) < 2:
        return pd.DataFrame()
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(df['resumo_processado'].fillna(''))
    similarities = cosine_similarity(tfidf_matrix[idx_referencia:idx_referencia+1], tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
    df_similar = df.iloc[similar_indices].copy()
    df_similar['similaridade'] = similarities[similar_indices]
    return df_similar

def prever_tendencias(df, anos_previsao=3):
    """Usa regressão linear simples para estimar tendências por tema."""
    resultados = []
    topicos = df['nome_topico'].dropna().unique()
    for tema in topicos:
        df_tema = df[df['nome_topico'] == tema].groupby('ano').size().reset_index(name='count')
        if len(df_tema) < 3:
            continue
        X = df_tema['ano'].values.reshape(-1, 1)
        y = df_tema['count'].values
        model = LinearRegression()
        model.fit(X, y)
        ultimo_ano = df_tema['ano'].max()
        anos_futuro = np.array([ultimo_ano + i for i in range(1, anos_previsao + 1)]).reshape(-1, 1)
        previsoes = model.predict(anos_futuro)
        previsoes = np.maximum(0, previsoes)
        ultimo_valor = int(df_tema.iloc[-1]['count'])
        previsao_media = float(previsoes.mean())
        percentual_mudanca = ((previsao_media - ultimo_valor) / ultimo_valor * 100) if ultimo_valor > 0 else 0
        score_tendencia = float(model.coef_[0])
        resultados.append({
            'tema': tema,
            'score_tendencia': score_tendencia,
            'ultimo_valor': ultimo_valor,
            'previsao_media': previsao_media,
            'percentual_mudanca': percentual_mudanca
        })
    return pd.DataFrame(resultados)

def extrair_termos_emergentes(df, top_n=20):
    """Identifica termos com maior crescimento entre dois períodos (antigo x recente)."""
    if len(df) < 50 or df['ano'].nunique() < 2:
        return pd.DataFrame()
    ano_corte = df['ano'].median()
    df_antigo = df[df['ano'] <= ano_corte]
    df_recente = df[df['ano'] > ano_corte]
    if df_antigo.empty or df_recente.empty:
        return pd.DataFrame()
    texto_antigo = ' '.join(df_antigo['resumo_processado'].dropna())
    texto_recente = ' '.join(df_recente['resumo_processado'].dropna())
    freq_antiga = Counter(texto_antigo.split())
    freq_recente = Counter(texto_recente.split())
    total_antigo = sum(freq_antiga.values()) + 1
    total_recente = sum(freq_recente.values()) + 1
    termos = []
    for termo, cont_recente in freq_recente.items():
        if cont_recente > 2:
            cont_antigo = freq_antiga.get(termo, 0)
            freq_rel_recente = cont_recente / total_recente
            freq_rel_antiga = cont_antigo / total_antigo
            if freq_rel_antiga > 0:
                crescimento_pct = ((freq_rel_recente - freq_rel_antiga) / freq_rel_antiga) * 100
            else:
                crescimento_pct = float('inf')
            termos.append({
                'termo': termo,
                'freq_antiga': cont_antigo,
                'freq_recente': cont_recente,
                'crescimento_pct': crescimento_pct
            })
    df_final = pd.DataFrame(termos)
    if df_final.empty:
        return df_final
    df_final = df_final.sort_values('crescimento_pct', ascending=False)
    return df_final.head(top_n)
