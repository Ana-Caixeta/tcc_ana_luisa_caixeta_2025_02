# -*- coding: utf-8 -*-
import streamlit as st
import plotly.express as px
from utilitarios import simplificar_topico, extract_keywords, metric_bold

def exibir(df):
    st.subheader("Análise Temática")
    df_temas = df.groupby('nome_topico').agg({
        'titulo': 'count',
        'instituicao': 'nunique',
        'curso': 'nunique'
    }).reset_index()
    df_temas.columns = ['tema', 'qtd_tccs', 'qtd_instituicoes', 'qtd_cursos']
    df_temas = df_temas.sort_values('qtd_tccs', ascending=False)
    df_temas['tema_simples'] = df_temas['tema'].apply(simplificar_topico)

    col1, col2, col3 = st.columns(3)
    with col1:
        metric_bold("Total de Temas", len(df_temas))
    with col2:
        tema_top = df_temas.iloc[0]['tema_simples'] if not df_temas.empty else 'N/A'
        metric_bold("Tema Mais Frequente", tema_top)
    with col3:
        metric_bold("Média TCCs/Tema", f"{df_temas['qtd_tccs'].mean():.1f}")

    st.markdown("---")
    st.subheader("Evolução Temporal dos Principais Temas")
    top_temas = df_temas.head(5)['tema'].tolist()
    df_tema_tempo = df[df['nome_topico'].isin(top_temas)].groupby(['ano', 'nome_topico']).size().reset_index(name='count')
    df_tema_tempo['tema_simples'] = df_tema_tempo['nome_topico'].apply(simplificar_topico)
    fig_tema_tempo = px.line(df_tema_tempo, x='ano', y='count', color='tema_simples', markers=True, labels={'count': 'TCCs', 'ano': 'Ano', 'tema_simples': 'Tema'})
    fig_tema_tempo.update_layout(height=400, hovermode='x unified')
    st.plotly_chart(fig_tema_tempo, use_container_width=True)

    st.markdown("---")
    st.subheader("Análise por Tema")
    tema_sel = st.selectbox("Selecione um tema", options=df_temas['tema'].tolist(), format_func=simplificar_topico)
    if tema_sel:
        df_tema_det = df[df['nome_topico'] == tema_sel]
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("TCCs", len(df_tema_det))
        with col_b:
            st.metric("Instituições", df_tema_det['instituicao'].nunique())
        with col_c:
            st.metric("Cursos", df_tema_det['curso'].nunique())

        st.write("**Top Palavras-Chave:**")
        keywords_tema = extract_keywords(df_tema_det['resumo_processado'], top_n=10)
        col1, col2 = st.columns(2)
        metade = len(keywords_tema[:10]) // 2
        col1_keywords = keywords_tema[:metade]
        col2_keywords = keywords_tema[metade:]
        with col1:
            for word, freq in col1_keywords:
                st.write(f"• {word}: {freq} ocorrências")
        with col2:
            for word, freq in col2_keywords:
                st.write(f"• {word}: {freq} ocorrências")


    if tema_sel:
        df_tema_det = df[df['nome_topico'] == tema_sel]
        st.subheader("Cursos Relacionados")
        cursos_tema = df_tema_det['curso'].value_counts().head(8).reset_index()
        cursos_tema.columns = ['curso', 'count']
        fig_cursos_tema = px.bar(cursos_tema, x='count', y='curso', orientation='h', labels={'count': 'TCCs', 'curso': 'Curso'})
        fig_cursos_tema.update_layout(
            height=350, 
            showlegend=False, 
            yaxis_autorange="reversed"
        )
        st.plotly_chart(fig_cursos_tema, use_container_width=True)

    st.markdown("---")
    st.subheader("Mapa de Calor: Temas × Cursos")
    top_cursos_heatmap = df['curso'].value_counts().head(6).index.tolist()
    top_temas_heatmap = df_temas.head(6)['tema'].tolist()
    df_heatmap = df[(df['curso'].isin(top_cursos_heatmap)) & (df['nome_topico'].isin(top_temas_heatmap))]
    if not df_heatmap.empty:
        pivot_table = df_heatmap.pivot_table(index='nome_topico', columns='curso', values='titulo', aggfunc='count', fill_value=0)
        pivot_table.index = pivot_table.index.map(simplificar_topico)
        fig_heatmap = px.imshow(pivot_table, labels=dict(x="Curso", y="Tema", color="TCCs"), aspect='auto')
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("Dados insuficientes para gerar o mapa de calor.")
