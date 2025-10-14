# -*- coding: utf-8 -*-
"""
Dashboard Avançado: Panorama Temático de TCCs na Rede Federal
Versão com múltiplas visões e análises aprofundadas
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# --- CORREÇÃO 1: Importar LinearRegression ---
from sklearn.linear_model import LinearRegression

# --- CONFIGURAÇÕES DA PÁGINA ---
st.set_page_config(
    page_title="Panorama Temático de TCCs",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CARREGAMENTO DE DADOS ---
@st.cache_data
def load_data():
    """Carrega o arquivo parquet gerado pelo preprocess.py"""
    try:
        df = pd.read_parquet("tccs_dashboard.parquet")
        required_cols = ['titulo', 'autores', 'ano', 'instituicao', 'resumo', 'resumo_processado', 'curso', 'nome_topico', 'orientador']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Colunas faltando: {missing_cols}")
            st.stop()
        
        # Converter ano para inteiro
        if df['ano'].dtype == 'object':
            df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
        df = df.dropna(subset=['ano'])
        df['ano'] = df['ano'].astype(int)
        
        return df
    except FileNotFoundError:
        st.error("❌ Arquivo 'tccs_dashboard.parquet' não encontrado.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao carregar dados: {e}")
        st.stop()

# --- FUNÇÕES AUXILIARES ---
def filter_data(df, instituicoes, anos, topicos, cursos=None):
    """Aplica filtros ao dataframe"""
    df_filtered = df.copy()
    
    if instituicoes:
        df_filtered = df_filtered[df_filtered['instituicao'].isin(instituicoes)]
    
    if anos:
        df_filtered = df_filtered[df_filtered['ano'].between(anos[0], anos[1])]
    
    if topicos:
        df_filtered = df_filtered[df_filtered['nome_topico'].isin(topicos)]
    
    if cursos:
        df_filtered = df_filtered[df_filtered['curso'].isin(cursos)]
    
    return df_filtered

def extract_keywords(texts, top_n=15):
    """Extrai palavras-chave mais frequentes"""
    all_words = []
    for text in texts:
        if isinstance(text, str):
            all_words.extend(text.split())
    word_freq = Counter(all_words)
    return word_freq.most_common(top_n)

def calcular_similaridade(df, idx_referencia, top_n=5):
    """Calcula TCCs similares usando TF-IDF e similaridade de cosseno"""
    if len(df) < 2:
        return pd.DataFrame()
    
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(df['resumo_processado'].fillna(''))
    
    similarities = cosine_similarity(tfidf_matrix[idx_referencia:idx_referencia+1], tfidf_matrix).flatten()
    similar_indices = similarities.argsort()[-top_n-1:-1][::-1]
    
    df_similar = df.iloc[similar_indices].copy()
    df_similar['similaridade'] = similarities[similar_indices]
    
    return df_similar

def simplificar_topico(nome_topico):
    """Remove 'Tópico X:' do nome"""
    return re.sub(r'Tópico \d+: ', '', str(nome_topico))

# --- CORREÇÃO 2: Definir a função prever_tendencias ---
def prever_tendencias(df, anos_previsao=3):
    """
    Usa Regressão Linear para prever a tendência de cada tema.
    Retorna um DataFrame com as previsões e um score de tendência.
    """
    resultados = []
    topicos = df['nome_topico'].dropna().unique()
    
    for tema in topicos:
        df_tema = df[df['nome_topico'] == tema].groupby('ano').size().reset_index(name='count')
        
        # Precisa de pelo menos 3 pontos para uma regressão mínima
        if len(df_tema) < 3:
            continue
            
        X = df_tema['ano'].values.reshape(-1, 1)
        y = df_tema['count'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Previsão
        ultimo_ano = df_tema['ano'].max()
        anos_futuro = np.array([ultimo_ano + i for i in range(1, anos_previsao + 1)]).reshape(-1, 1)
        previsoes = model.predict(anos_futuro)
        previsoes = np.maximum(0, previsoes) # Evitar previsões negativas
        
        ultimo_valor = df_tema.iloc[-1]['count']
        previsao_media = previsoes.mean()
        
        # Calcular mudança percentual
        if ultimo_valor > 0:
            percentual_mudanca = ((previsao_media - ultimo_valor) / ultimo_valor) * 100
        else:
            percentual_mudanca = 0
            
        # O coeficiente angular da reta (slope) é um ótimo score de tendência
        score_tendencia = model.coef_[0]
        
        resultados.append({
            'tema': tema,
            'score_tendencia': score_tendencia,
            'ultimo_valor': ultimo_valor,
            'previsao_media': previsao_media,
            'percentual_mudanca': percentual_mudanca
        })
        
    return pd.DataFrame(resultados)

# --- CORREÇÃO 3: Definir a função extrair_termos_emergentes ---
def extrair_termos_emergentes(df, top_n=20):
    """
    Identifica termos que se tornaram mais frequentes recentemente.
    Divide o período em dois e compara a frequência relativa das palavras.
    """
    if len(df) < 50 or df['ano'].nunique() < 2:
        return pd.DataFrame()
        
    ano_corte = df['ano'].median()
    df_antigo = df[df['ano'] <= ano_corte]
    df_recente = df[df['ano'] > ano_corte]
    
    if df_antigo.empty or df_recente.empty:
        return pd.DataFrame()

    # Concatenar todos os resumos processados de cada período
    texto_antigo = ' '.join(df_antigo['resumo_processado'].dropna())
    texto_recente = ' '.join(df_recente['resumo_processado'].dropna())
    
    # Contar frequência das palavras
    freq_antiga = Counter(texto_antigo.split())
    freq_recente = Counter(texto_recente.split())
    
    total_antigo = sum(freq_antiga.values()) + 1 # +1 para evitar divisão por zero
    total_recente = sum(freq_recente.values()) + 1
    
    termos_crescimento = []
    
    for termo, cont_recente in freq_recente.items():
        # Considerar apenas termos com alguma relevância
        if cont_recente > 2: 
            cont_antigo = freq_antiga.get(termo, 0)
            
            # Usar frequência relativa para normalizar
            freq_rel_recente = cont_recente / total_recente
            freq_rel_antiga = cont_antigo / total_antigo
            
            if freq_rel_antiga > 0:
                crescimento_pct = ((freq_rel_recente - freq_rel_antiga) / freq_rel_antiga) * 100
            else:
                crescimento_pct = float('inf') # Termo novo
                
            termos_crescimento.append({
                'termo': termo,
                'freq_antiga': cont_antigo,
                'freq_recente': cont_recente,
                'crescimento_pct': crescimento_pct
            })

    df_final = pd.DataFrame(termos_crescimento)
    df_final = df_final.sort_values('crescimento_pct', ascending=False)
    
    return df_final.head(top_n)

# --- CSS PERSONALIZADO ---
st.markdown("""
    <style>
    .stApp {
        background-color: #F6F6F6;
    }
    
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
    }
    
    .main-header {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        padding: 25px;
        border-radius: 10px;
        color: white !important;
        margin-bottom: 20px;
        text-align: center;
    }
    .main-header h1 {
        margin: 0;
        font-size: 2.2em;
        color: white !important;
    }
    
    /* Menu superior */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #FFFFFF;
        padding: 10px;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        background-color: #F6F6F6;
        border-radius: 8px;
        color: #1A1A1A;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4A90E2 !important;
        color: white !important;
    }
    
    /* Cards de métricas */
    [data-testid="stMetricValue"] {
        font-size: 2em;
        color: #1A1A1A;
    }
    
    /* Tabelas */
    [data-testid="stDataFrame"] {
        background-color: #FFFFFF;
    }
    </style>
""", unsafe_allow_html=True)

# --- CARREGAMENTO ---
df = load_data()

# --- HEADER ---
st.markdown("""
    <div class="main-header">
        <h1>📚 Panorama Temático de TCCs na Rede Federal</h1>
        <p style='margin: 5px 0 0 0; font-size: 1.1em;'>Análise Inteligente de Trabalhos de Conclusão de Curso</p>
    </div>
""", unsafe_allow_html=True)

# --- MENU SUPERIOR COM ABAS ---
# Adicionada a aba de Tendências que estava faltando no seu código original
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🏠 Visão Geral", 
    "👨‍🏫 Orientadores", 
    "🏛️ Instituições", 
    "🎯 Temáticas",
    "🔍 Busca Avançada",
    "📈 Tendências" # A lógica já existia, mas a aba não estava na lista
])

# --- SIDEBAR - FILTROS GLOBAIS ---
with st.sidebar:
    st.header("🔍 FILTROS GLOBAIS")
    
    st.subheader("📅 Período")
    ano_min = int(df['ano'].min())
    ano_max = int(df['ano'].max())
    anos_selecionados = st.slider(
        "Intervalo de anos",
        min_value=ano_min,
        max_value=ano_max,
        value=(ano_min, ano_max),
        label_visibility="collapsed"
    )
    
    st.subheader("🏛️ Instituições")
    instituicoes_disponiveis = sorted(df['instituicao'].dropna().unique())
    instituicoes_selecionadas = st.multiselect(
        "Selecione",
        options=instituicoes_disponiveis,
        default=[],
        label_visibility="collapsed"
    )
    
    st.subheader("📚 Cursos")
    cursos_disponiveis = sorted(df['curso'].dropna().unique())
    cursos_selecionados = st.multiselect(
        "Selecione",
        options=cursos_disponiveis,
        default=[],
        label_visibility="collapsed"
    )
    
    st.subheader("🎯 Temas")
    topicos_disponiveis = sorted(df['nome_topico'].dropna().unique())
    topicos_selecionados = st.multiselect(
        "Selecione",
        options=topicos_disponiveis,
        default=[],
        label_visibility="collapsed"
    )

# Aplicar filtros globais
df_filtered = filter_data(df, instituicoes_selecionadas, anos_selecionados, topicos_selecionados, cursos_selecionados)

if df_filtered.empty:
    st.warning("⚠️ Nenhum dado encontrado para os filtros selecionados. Por favor, ajuste os filtros.")
    st.stop()

# ==============================================
# TAB 1: VISÃO GERAL
# ==============================================
with tab1:
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total de TCCs", f"{len(df_filtered):,}".replace(",", "."))
    with col2:
        st.metric("🏛️ Instituições", df_filtered['instituicao'].nunique())
    with col3:
        st.metric("👨‍🏫 Orientadores", df_filtered['orientador'].nunique())
    with col4:
        st.metric("🎯 Temas", df_filtered['nome_topico'].nunique())
    
    st.markdown("---")
    
    # Gráficos principais
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("📈 Produção Anual de TCCs")
        df_ano = df_filtered.groupby('ano').size().reset_index(name='count')
        fig_ano = px.bar(df_ano, x='ano', y='count', 
                         labels={'count': 'Quantidade', 'ano': 'Ano'},
                         color_discrete_sequence=['#4A90E2'])
        fig_ano.update_layout(height=400, showlegend=False, yaxis_title="Quantidade de TCCs")
        st.plotly_chart(fig_ano, use_container_width=True)
    
    with col_right:
        st.subheader("🎯 Distribuição por Tema")
        df_topicos = df_filtered['nome_topico'].value_counts().head(8).reset_index()
        df_topicos.columns = ['tema', 'count']
        df_topicos['tema_simples'] = df_topicos['tema'].apply(simplificar_topico)
        
        fig_pizza = px.pie(df_topicos, values='count', names='tema_simples', hole=0.4)
        fig_pizza.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pizza, use_container_width=True)
    
    st.markdown("---")
    
    # Top instituições e cursos
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🏆 Top 5 Instituições")
        top_inst = df_filtered['instituicao'].value_counts().head(5).reset_index()
        top_inst.columns = ['Instituição', 'TCCs']
        st.dataframe(top_inst, hide_index=True, use_container_width=True)
    
    with col2:
        st.subheader("🏆 Top 5 Cursos")
        top_cursos = df_filtered['curso'].value_counts().head(5).reset_index()
        top_cursos.columns = ['Curso', 'TCCs']
        st.dataframe(top_cursos, hide_index=True, use_container_width=True)

# ==============================================
# TAB 2: ORIENTADORES
# ==============================================
with tab2:
    st.subheader("👨‍🏫 Análise de Orientadores")
    
    # KPIs de orientadores
    df_orient = df_filtered.groupby('orientador').agg({
        'titulo': 'count',
        'nome_topico': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
    }).reset_index()
    df_orient.columns = ['orientador', 'qtd_orientacoes', 'tema_principal']
    df_orient = df_orient.sort_values('qtd_orientacoes', ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Orientadores", len(df_orient))
    with col2:
        st.metric("Média Orientações/Prof", f"{df_orient['qtd_orientacoes'].mean():.1f}")
    with col3:
        max_orientacoes = int(df_orient['qtd_orientacoes'].max()) if not df_orient.empty else 0
        st.metric("Máx Orientações", max_orientacoes)
    with col4:
        orientador_top = df_orient.iloc[0]['orientador'] if not df_orient.empty else 'N/A'
        st.metric("Orientador Mais Ativo", orientador_top if len(str(orientador_top)) < 20 else str(orientador_top)[:17] + "...")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("📊 Top 15 Orientadores")
        top_orient = df_orient.head(15).copy().sort_values('qtd_orientacoes', ascending=True)
        
        fig_orient = px.bar(top_orient, x='qtd_orientacoes', y='orientador',
                            orientation='h',
                            labels={'qtd_orientacoes': 'Orientações', 'orientador': 'Orientador'},
                            color='qtd_orientacoes',
                            color_continuous_scale='Blues')
        fig_orient.update_layout(height=600, showlegend=False, yaxis_title="")
        st.plotly_chart(fig_orient, use_container_width=True)
    
    with col_right:
        st.subheader("🔍 Detalhes por Orientador")
        
        orientador_selecionado = st.selectbox(
            "Escolha um orientador",
            options=df_orient['orientador'].tolist()
        )
        
        if orientador_selecionado:
            df_prof = df_filtered[df_filtered['orientador'] == orientador_selecionado]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Total Orientações", len(df_prof))
            with col_b:
                anos_atuacao = df_prof['ano'].max() - df_prof['ano'].min() + 1 if not df_prof.empty else 0
                st.metric("Anos de Atuação", anos_atuacao)
            
            st.write("**Temas de Atuação:**")
            temas_prof = df_prof['nome_topico'].value_counts().head(5)
            for tema, count in temas_prof.items():
                st.write(f"• {simplificar_topico(tema)}: {count} TCCs")
            
            st.write("**Evolução Temporal:**")
            df_prof_tempo = df_prof.groupby('ano').size().reset_index(name='count')
            fig_prof_tempo = px.line(df_prof_tempo, x='ano', y='count', markers=True,
                                     labels={'count': 'Orientações', 'ano': 'Ano'})
            fig_prof_tempo.update_layout(height=250, showlegend=False)
            st.plotly_chart(fig_prof_tempo, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("📋 Ranking Completo de Orientadores")
    df_orient_display = df_orient.copy()
    df_orient_display['tema_simples'] = df_orient_display['tema_principal'].apply(simplificar_topico)
    df_orient_display = df_orient_display[['orientador', 'qtd_orientacoes', 'tema_simples']]
    df_orient_display.columns = ['Orientador', 'Orientações', 'Tema Principal']
    
    st.dataframe(df_orient_display, hide_index=True, use_container_width=True, height=400)

# ==============================================
# TAB 3: INSTITUIÇÕES
# ==============================================
with tab3:
    st.subheader("🏛️ Análise Institucional")
    
    df_inst = df_filtered.groupby('instituicao').agg({
        'titulo': 'count',
        'orientador': 'nunique',
        'curso': 'nunique',
        'nome_topico': 'nunique'
    }).reset_index()
    df_inst.columns = ['instituicao', 'qtd_tccs', 'qtd_orientadores', 'qtd_cursos', 'qtd_temas']
    df_inst = df_inst.sort_values('qtd_tccs', ascending=False)
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Instituições", len(df_inst))
    with col2:
        st.metric("Média TCCs/Instituição", f"{df_inst['qtd_tccs'].mean():.1f}")
    with col3:
        st.metric("Média Orientadores/Inst", f"{df_inst['qtd_orientadores'].mean():.1f}")
    with col4:
        st.metric("Diversidade Temática Média", f"{df_inst['qtd_temas'].mean():.1f}")
    
    st.markdown("---")
    
    st.subheader("📊 Comparativo entre Instituições")
    top_inst_chart = df_inst.head(15)
    
    fig_inst = go.Figure()
    fig_inst.add_trace(go.Bar(name='TCCs', x=top_inst_chart['instituicao'], y=top_inst_chart['qtd_tccs']))
    fig_inst.add_trace(go.Bar(name='Orientadores', x=top_inst_chart['instituicao'], y=top_inst_chart['qtd_orientadores']))
    fig_inst.add_trace(go.Bar(name='Cursos', x=top_inst_chart['instituicao'], y=top_inst_chart['qtd_cursos']))
    fig_inst.update_layout(barmode='group', height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_inst, use_container_width=True)
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("🔍 Análise Detalhada")
        instituicao_sel = st.selectbox(
            "Selecione uma instituição",
            options=df_inst['instituicao'].tolist()
        )
        
        if instituicao_sel:
            df_inst_det = df_filtered[df_filtered['instituicao'] == instituicao_sel]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("TCCs", len(df_inst_det))
            with col_b:
                st.metric("Orientadores", df_inst_det['orientador'].nunique())
            with col_c:
                st.metric("Cursos", df_inst_det['curso'].nunique())
            
            st.write("**Top 5 Cursos:**")
            top_cursos_inst = df_inst_det['curso'].value_counts().head(5)
            for curso, count in top_cursos_inst.items():
                st.write(f"• {curso}: {count} TCCs")
    
    with col_right:
        if instituicao_sel:
            df_inst_det = df_filtered[df_filtered['instituicao'] == instituicao_sel]
            st.subheader("📈 Evolução Temporal")
            df_inst_tempo = df_inst_det.groupby('ano').size().reset_index(name='count')
            fig_inst_tempo = px.area(df_inst_tempo, x='ano', y='count',
                                     labels={'count': 'TCCs', 'ano': 'Ano'},
                                     color_discrete_sequence=['#4A90E2'])
            fig_inst_tempo.update_layout(height=300)
            st.plotly_chart(fig_inst_tempo, use_container_width=True)
            
            st.subheader("🎯 Distribuição Temática")
            temas_inst = df_inst_det['nome_topico'].value_counts().head(5).reset_index()
            temas_inst.columns = ['tema', 'count']
            temas_inst['tema_simples'] = temas_inst['tema'].apply(simplificar_topico)
            
            fig_temas_inst = px.bar(temas_inst, x='tema_simples', y='count',
                                    labels={'count': 'TCCs', 'tema_simples': 'Tema'},
                                    color_discrete_sequence=['#357ABD'])
            fig_temas_inst.update_layout(height=300, showlegend=False)
            st.plotly_chart(fig_temas_inst, use_container_width=True)

# ==============================================
# TAB 4: TEMÁTICAS
# ==============================================
with tab4:
    st.subheader("🎯 Análise Temática")
    
    df_temas = df_filtered.groupby('nome_topico').agg({
        'titulo': 'count',
        'instituicao': 'nunique',
        'curso': 'nunique'
    }).reset_index()
    df_temas.columns = ['tema', 'qtd_tccs', 'qtd_instituicoes', 'qtd_cursos']
    df_temas = df_temas.sort_values('qtd_tccs', ascending=False)
    df_temas['tema_simples'] = df_temas['tema'].apply(simplificar_topico)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total de Temas", len(df_temas))
    with col2:
        tema_top = df_temas.iloc[0]['tema_simples'] if not df_temas.empty else 'N/A'
        st.metric("Tema Mais Frequente", tema_top if len(tema_top) < 25 else tema_top[:22] + "...")
    with col3:
        st.metric("Média TCCs/Tema", f"{df_temas['qtd_tccs'].mean():.1f}")
    
    st.markdown("---")
    
    st.subheader("📈 Evolução Temporal dos Principais Temas")
    
    top_temas = df_temas.head(5)['tema'].tolist()
    df_tema_tempo = df_filtered[df_filtered['nome_topico'].isin(top_temas)].groupby(['ano', 'nome_topico']).size().reset_index(name='count')
    df_tema_tempo['tema_simples'] = df_tema_tempo['nome_topico'].apply(simplificar_topico)
    
    fig_tema_tempo = px.line(df_tema_tempo, x='ano', y='count', color='tema_simples',
                             markers=True,
                             labels={'count': 'TCCs', 'ano': 'Ano', 'tema_simples': 'Tema'})
    fig_tema_tempo.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_tema_tempo, use_container_width=True)
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        st.subheader("🔍 Análise por Tema")
        tema_sel = st.selectbox(
            "Selecione um tema",
            options=df_temas['tema'].tolist(),
            format_func=simplificar_topico
        )
        
        if tema_sel:
            df_tema_det = df_filtered[df_filtered['nome_topico'] == tema_sel]
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("TCCs", len(df_tema_det))
            with col_b:
                st.metric("Instituições", df_tema_det['instituicao'].nunique())
            with col_c:
                st.metric("Cursos", df_tema_det['curso'].nunique())
            
            st.write("**Top Palavras-Chave:**")
            keywords_tema = extract_keywords(df_tema_det['resumo_processado'], top_n=10)
            for word, freq in keywords_tema[:5]:
                st.write(f"• {word.capitalize()}: {freq} ocorrências")
    
    with col_right:
        if tema_sel:
            df_tema_det = df_filtered[df_filtered['nome_topico'] == tema_sel]
            st.subheader("📊 Cursos Relacionados")
            cursos_tema = df_tema_det['curso'].value_counts().head(8).reset_index()
            cursos_tema.columns = ['curso', 'count']
            
            fig_cursos_tema = px.bar(cursos_tema, x='count', y='curso',
                                     orientation='h',
                                     labels={'count': 'TCCs', 'curso': 'Curso'},
                                     color_discrete_sequence=['#4A90E2'])
            fig_cursos_tema.update_layout(height=350, showlegend=False)
            st.plotly_chart(fig_cursos_tema, use_container_width=True)
    
    st.markdown("---")
    
    st.subheader("🔥 Mapa de Calor: Temas × Cursos")
    
    top_cursos_heatmap = df_filtered['curso'].value_counts().head(6).index.tolist()
    top_temas_heatmap = df_temas.head(6)['tema'].tolist()
    
    df_heatmap = df_filtered[
        (df_filtered['curso'].isin(top_cursos_heatmap)) &
        (df_filtered['nome_topico'].isin(top_temas_heatmap))
    ]
    
    if not df_heatmap.empty:
        pivot_table = df_heatmap.pivot_table(
            index='nome_topico',
            columns='curso',
            values='titulo',
            aggfunc='count',
            fill_value=0
        )
        
        pivot_table.index = pivot_table.index.map(simplificar_topico)
        
        fig_heatmap = px.imshow(pivot_table,
                                labels=dict(x="Curso", y="Tema", color="TCCs"),
                                color_continuous_scale='Blues',
                                aspect='auto')
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar o mapa de calor.")

# ==============================================
# TAB 5: BUSCA AVANÇADA
# ==============================================
with tab5:
    st.subheader("🔍 Busca Avançada e Similaridade")
    
    col1, col2 = st.columns([3, 1])
    with col1:
        busca = st.text_input("🔎 Buscar em títulos e resumos", "")
    with col2:
        limite_resultados = st.number_input("Limite", min_value=5, max_value=100, value=20, step=5)
    
    if busca:
        mask = (
            df_filtered['titulo'].str.contains(busca, case=False, na=False) |
            df_filtered['resumo'].str.contains(busca, case=False, na=False)
        )
        df_busca = df_filtered[mask].head(limite_resultados)
        
        st.success(f"✅ Encontrados {len(df_busca)} resultados")
        
        for idx, row in df_busca.iterrows():
            with st.expander(f"📄 {row['titulo']}"):
                col_a, col_b = st.columns([2, 1])
                
                with col_a:
                    st.write(f"**Autores:** {row['autores']}")
                    st.write(f"**Orientador:** {row['orientador']}")
                    st.write(f"**Ano:** {row['ano']}")
                    st.write(f"**Curso:** {row['curso']}")
                
                with col_b:
                    st.write(f"**Instituição:** {row['instituicao']}")
                    st.write(f"**Tema:** {simplificar_topico(row['nome_topico'])}")
                
                st.write("**Resumo:**")
                resumo_preview = str(row['resumo'])[:300] + "..." if len(str(row['resumo'])) > 300 else str(row['resumo'])
                st.write(resumo_preview)
                
                if st.button(f"🔗 Encontrar TCCs similares", key=f"sim_{idx}"):
                    st.session_state[f'buscar_similar_{idx}'] = True
                
                if st.session_state.get(f'buscar_similar_{idx}', False):
                    with st.spinner("Calculando similaridade..."):
                        df_filtered_reset = df_filtered.reset_index(drop=True)
                        idx_matches = df_filtered_reset[df_filtered_reset['titulo'] == row['titulo']].index
                        if not idx_matches.empty:
                            idx_relative = idx_matches[0]
                            df_similar = calcular_similaridade(df_filtered_reset, idx_relative, top_n=5)
                            
                            if not df_similar.empty:
                                st.write("**📊 TCCs Similares:**")
                                for _, sim_row in df_similar.iterrows():
                                    similarity_pct = sim_row['similaridade'] * 100
                                    st.write(f"• **{sim_row['titulo']}** (Similaridade: {similarity_pct:.1f}%)")
                                    st.write(f"  ↳ {sim_row['autores']} - {sim_row['ano']}")
                            else:
                                st.info("Nenhum TCC similar encontrado.")
                        else:
                            st.warning("Não foi possível localizar o TCC de referência para calcular a similaridade.")
    
    else:
        st.info("💡 Digite um termo para buscar em títulos e resumos dos TCCs")
    
    st.markdown("---")
    
    st.subheader("🔗 Análise de Similaridade entre TCCs")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        titulos_disponiveis = df_filtered['titulo'].tolist()
        tcc_selecionado = st.selectbox(
            "Selecione um TCC para encontrar trabalhos similares",
            options=titulos_disponiveis,
            key="similarity_selector"
        )
    
    with col2:
        num_similares = st.number_input("Quantidade", min_value=3, max_value=20, value=5, step=1, key="num_sim")
    
    if tcc_selecionado and st.button("🔍 Buscar TCCs Similares", key="btn_similarity"):
        with st.spinner("Analisando similaridade..."):
            df_filtered_reset = df_filtered.reset_index(drop=True)
            idx_selecionado = df_filtered_reset[df_filtered_reset['titulo'] == tcc_selecionado].index[0]
            
            tcc_info = df_filtered_reset.iloc[idx_selecionado]
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**📄 TCC de Referência:**")
                st.write(f"**Título:** {tcc_info['titulo']}")
                st.write(f"**Autores:** {tcc_info['autores']}")
                st.write(f"**Ano:** {tcc_info['ano']}")
            
            with col_b:
                st.write(f"**Instituição:** {tcc_info['instituicao']}")
                st.write(f"**Curso:** {tcc_info['curso']}")
                st.write(f"**Tema:** {simplificar_topico(tcc_info['nome_topico'])}")
            
            df_similar = calcular_similaridade(df_filtered_reset, idx_selecionado, top_n=num_similares)
            
            if not df_similar.empty:
                st.markdown("---")
                st.write(f"**🎯 Top {num_similares} TCCs Mais Similares:**")
                
                for i, (_, sim_row) in enumerate(df_similar.iterrows(), 1):
                    similarity_pct = sim_row['similaridade'] * 100
                    
                    with st.expander(f"#{i} - {sim_row['titulo']} ({similarity_pct:.1f}% similar)"):
                        col_x, col_y = st.columns(2)
                        with col_x:
                            st.write(f"**Autores:** {sim_row['autores']}")
                            st.write(f"**Orientador:** {sim_row['orientador']}")
                            st.write(f"**Ano:** {sim_row['ano']}")
                        with col_y:
                            st.write(f"**Instituição:** {sim_row['instituicao']}")
                            st.write(f"**Curso:** {sim_row['curso']}")
                            st.write(f"**Tema:** {simplificar_topico(sim_row['nome_topico'])}")
                        
                        st.write("**Resumo:**")
                        resumo_sim = str(sim_row['resumo'])[:250] + "..." if len(str(sim_row['resumo'])) > 250 else str(sim_row['resumo'])
                        st.write(resumo_sim)
                
                st.markdown("---")
                st.write("**📊 Visualização de Similaridade:**")
                
                df_sim_viz = df_similar.copy().sort_values('similaridade', ascending=True)
                df_sim_viz['titulo_curto'] = df_sim_viz['titulo'].apply(lambda x: x[:40] + "..." if len(x) > 40 else x)
                df_sim_viz['similaridade_pct'] = df_sim_viz['similaridade'] * 100
                
                fig_sim = px.bar(df_sim_viz, x='similaridade_pct', y='titulo_curto', orientation='h',
                                 labels={'similaridade_pct': 'Similaridade (%)', 'titulo_curto': 'TCC'},
                                 color='similaridade_pct', color_continuous_scale='Viridis')
                fig_sim.update_layout(height=400, showlegend=False, yaxis_title="")
                st.plotly_chart(fig_sim, use_container_width=True)
            else:
                st.warning("⚠️ Não foi possível calcular similaridades.")

# ==============================================
# TAB 6: TENDÊNCIAS E PREVISÕES
# ==============================================
with tab6:
    st.subheader("📈 Análise de Tendências e Previsões com Machine Learning")
    
    st.info("🤖 Esta análise utiliza modelos de Machine Learning para identificar tendências e prever temas em ascensão")
    
    col_config1, col_config2 = st.columns([3, 1])
    with col_config1:
        st.write("**Configurações de Análise:**")
    with col_config2:
        anos_previsao = st.selectbox("Anos para previsão", [2, 3, 4, 5], index=1)
    
    if st.button("🚀 Executar Análise de Tendências", type="primary"):
        with st.spinner("🔄 Processando dados e treinando modelos..."):
            
            # 1. ANÁLISE DE TENDÊNCIAS POR TEMA
            st.markdown("---")
            st.subheader("📊 Tendências por Tema")
            
            df_tendencias = prever_tendencias(df_filtered, anos_previsao=anos_previsao)
            
            if not df_tendencias.empty:
                df_tendencias['classificacao'] = df_tendencias['score_tendencia'].apply(
                    lambda x: '🚀 Alta Crescimento' if x > 2 else 
                              ('📈 Crescimento Moderado' if x > 0 else 
                               ('📉 Declínio Moderado' if x > -2 else '⚠️ Forte Declínio'))
                )
                df_tendencias['tema_simples'] = df_tendencias['tema'].apply(simplificar_topico)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    temas_crescimento = len(df_tendencias[df_tendencias['score_tendencia'] > 0])
                    st.metric("Temas em Crescimento", temas_crescimento)
                with col2:
                    temas_declinio = len(df_tendencias[df_tendencias['score_tendencia'] < 0])
                    st.metric("Temas em Declínio", temas_declinio)
                with col3:
                    melhor_tema = df_tendencias.nlargest(1, 'score_tendencia').iloc[0]['tema_simples']
                    st.metric("Maior Potencial", melhor_tema[:20] + "..." if len(melhor_tema) > 20 else melhor_tema)
                with col4:
                    crescimento_medio = df_tendencias['percentual_mudanca'].mean()
                    st.metric("Crescimento Médio", f"{crescimento_medio:.1f}%")
                
                st.write("**🚀 Top 10 Temas com Maior Potencial de Crescimento:**")
                top_crescimento = df_tendencias.nlargest(10, 'score_tendencia').sort_values('score_tendencia', ascending=True)
                
                colors = ['#00CC96' if x > 2 else '#FFA15A' if x > 0 else '#EF553B' for x in top_crescimento['score_tendencia']]
                
                fig_tendencias = go.Figure(go.Bar(
                    y=top_crescimento['tema_simples'],
                    x=top_crescimento['score_tendencia'],
                    orientation='h',
                    marker=dict(color=colors),
                    text=top_crescimento['score_tendencia'].round(2),
                    textposition='outside'
                ))
                fig_tendencias.update_layout(height=500, xaxis_title="Score de Tendência", yaxis_title="", showlegend=False)
                st.plotly_chart(fig_tendencias, use_container_width=True)
                
                st.write("**📋 Análise Detalhada de Tendências:**")
                df_display = df_tendencias[['tema_simples', 'ultimo_valor', 'previsao_media', 'percentual_mudanca', 'classificacao']].copy()
                df_display.columns = ['Tema', 'TCCs Atuais', 'Previsão Média', 'Mudança %', 'Tendência']
                df_display = df_display.sort_values('Mudança %', ascending=False)
                df_display['TCCs Atuais'] = df_display['TCCs Atuais'].round(0).astype(int)
                df_display['Previsão Média'] = df_display['Previsão Média'].round(1)
                df_display['Mudança %'] = df_display['Mudança %'].round(1)
                st.dataframe(df_display, hide_index=True, use_container_width=True, height=400)
            else:
                st.warning("⚠️ Dados insuficientes para análise de tendências. Ajuste os filtros.")
            
            # 2. TERMOS EMERGENTES
            st.markdown("---")
            st.subheader("🔥 Termos e Conceitos Emergentes")
            st.write("Análise de palavras-chave que estão ganhando popularidade nos TCCs mais recentes")
            
            df_emergentes = extrair_termos_emergentes(df_filtered, top_n=15)
            
            if not df_emergentes.empty:
                col_left, col_right = st.columns([1, 1])
                with col_left:
                    st.write("**📈 Top 15 Termos em Ascensão:**")
                    df_emergentes_chart = df_emergentes.sort_values('crescimento_pct', ascending=True)
                    fig_emergentes = px.bar(
                        df_emergentes_chart.head(15), x='crescimento_pct', y='termo', orientation='h',
                        color='crescimento_pct', color_continuous_scale='Viridis',
                        labels={'crescimento_pct': 'Crescimento (%)', 'termo': 'Termo'}
                    )
                    fig_emergentes.update_layout(height=500, showlegend=False, yaxis_title="")
                    st.plotly_chart(fig_emergentes, use_container_width=True)
                
                with col_right:
                    st.write("**📊 Detalhamento dos Termos:**")
                    df_emerg_display = df_emergentes[['termo', 'freq_antiga', 'freq_recente', 'crescimento_pct']].copy()
                    df_emerg_display.columns = ['Termo', 'Freq. Antiga', 'Freq. Recente', 'Crescimento %']
                    df_emerg_display['Crescimento %'] = df_emerg_display['Crescimento %'].round(1)
                    st.dataframe(df_emerg_display, hide_index=True, use_container_width=True, height=500)
            else:
                st.warning("⚠️ Não foi possível identificar termos emergentes. Verifique os dados.")
            
            # 3. PREVISÃO TEMPORAL
            st.markdown("---")
            st.subheader("🔮 Previsão de Produção por Tema")
            st.write(f"Previsão da quantidade de TCCs para os próximos {anos_previsao} anos")
            
            top_5_temas = df_filtered['nome_topico'].value_counts().head(5).index.tolist()
            
            if top_5_temas:
                tema_viz = st.selectbox(
                    "Selecione um tema para visualizar a previsão",
                    options=top_5_temas,
                    format_func=simplificar_topico
                )
                
                if tema_viz:
                    df_tema_hist = df_filtered[df_filtered['nome_topico'] == tema_viz].groupby('ano').size().reset_index(name='count')
                    
                    if len(df_tema_hist) >= 2:
                        X_hist = df_tema_hist['ano'].values.reshape(-1, 1)
                        y_hist = df_tema_hist['count'].values
                        
                        model = LinearRegression()
                        model.fit(X_hist, y_hist)
                        
                        ano_max_hist = df_filtered['ano'].max()
                        anos_futuro = np.array([ano_max_hist + i for i in range(1, anos_previsao + 1)]).reshape(-1, 1)
                        previsoes = model.predict(anos_futuro)
                        previsoes = np.maximum(previsoes, 0)
                        
                        df_previsao = pd.DataFrame({'ano': anos_futuro.flatten(), 'count': previsoes, 'tipo': 'Previsão'})
                        df_historico = df_tema_hist.copy()
                        df_historico['tipo'] = 'Histórico'
                        
                        df_viz = pd.concat([df_historico, df_previsao])
                        
                        fig_previsao = px.line(
                            df_viz, x='ano', y='count', color='tipo', markers=True,
                            labels={'count': 'Quantidade de TCCs', 'ano': 'Ano'},
                            color_discrete_map={'Histórico': '#4A90E2', 'Previsão': '#FF6B6B'}
                        )
                        fig_previsao.update_layout(height=400)
                        st.plotly_chart(fig_previsao, use_container_width=True)
                        
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Último Ano Real", f"{int(df_tema_hist.iloc[-1]['count'])} TCCs")
                        with col_b:
                            st.metric(f"Previsão para {int(anos_futuro[0][0])}", f"{int(previsoes[0])} TCCs")
                        with col_c:
                            variacao = ((previsoes[0] - df_tema_hist.iloc[-1]['count']) / df_tema_hist.iloc[-1]['count'] * 100) if df_tema_hist.iloc[-1]['count'] > 0 else 0
                            st.metric("Variação Prevista", f"{variacao:.1f}%")
                    else:
                        st.warning("Dados insuficientes para este tema para gerar uma previsão.")
            else:
                st.warning("Nenhum tema com dados suficientes para previsão.")
    else:
        st.info("👆 Clique no botão acima para executar a análise de tendências com Machine Learning")
        
        st.markdown("---")
        st.subheader("🔬 Metodologia")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**📊 Regressão Linear**")
            st.write("Análise de tendências temporais para cada tema")
        with col2:
            st.write("**📈 Análise de Crescimento**")
            st.write("Comparação entre períodos para identificar termos emergentes")
        with col3:
            st.write("**🔮 Previsão**")
            st.write(f"Projeção de TCCs para os próximos anos")

# --- RODAPÉ ---
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 20px;'>
        <p><strong>Dashboard Avançado de TCCs - Rede Federal</strong></p>
        <p>Desenvolvido com Streamlit • Dados processados com NLP e Machine Learning</p>
    </div>
""", unsafe_allow_html=True)
