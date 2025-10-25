# -*- coding: utf-8 -*-
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from utilitarios import simplificar_topico, metric_bold

def exibir(df):
    st.subheader("Análise Institucional")
    df_inst = df.groupby('instituicao').agg({
        'titulo': 'count',
        'orientador': 'nunique',
        'curso': 'nunique',
        'nome_topico': 'nunique'
    }).reset_index()
    df_inst.columns = ['instituicao', 'qtd_tccs', 'qtd_orientadores', 'qtd_cursos', 'qtd_temas']
    df_inst = df_inst.sort_values('qtd_tccs', ascending=False)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_bold("Total Instituições", len(df_inst))
    with col2:
        metric_bold("Média TCCs/Instituição", f"{df_inst['qtd_tccs'].mean():.1f}")
    with col3:
        metric_bold("Média Orientadores/Inst", f"{df_inst['qtd_orientadores'].mean():.1f}")
    with col4:
        metric_bold("Diversidade Temática Média", f"{df_inst['qtd_temas'].mean():.1f}")

    st.markdown("---")
    st.subheader("Comparativo entre Instituições")
    top_inst_chart = df_inst.head(15)
    fig_inst = go.Figure()
    fig_inst.add_trace(go.Bar(name='TCCs', x=top_inst_chart['instituicao'], y=top_inst_chart['qtd_tccs']))
    fig_inst.add_trace(go.Bar(name='Orientadores', x=top_inst_chart['instituicao'], y=top_inst_chart['qtd_orientadores']))
    fig_inst.add_trace(go.Bar(name='Cursos', x=top_inst_chart['instituicao'], y=top_inst_chart['qtd_cursos']))
    fig_inst.update_layout(barmode='group', height=400, xaxis_tickangle=-45)
    st.plotly_chart(fig_inst, config = {'responsive': True})

    st.markdown("---")
    st.subheader("Análise Detalhada")
    instituicao_sel = st.selectbox("Selecione uma instituição", options=df_inst['instituicao'].tolist())
    if instituicao_sel:
        df_inst_det = df[df['instituicao'] == instituicao_sel]
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

    if instituicao_sel:
        df_inst_det = df[df['instituicao'] == instituicao_sel]
        st.subheader("Evolução Temporal")
        df_inst_tempo = df_inst_det.groupby('ano').size().reset_index(name='count')
        fig_inst_tempo = px.area(df_inst_tempo, x='ano', y='count', labels={'count': 'TCCs', 'ano': 'Ano'})
        fig_inst_tempo.update_layout(height=300)
        st.plotly_chart(fig_inst_tempo, config = {'responsive': True})

        st.subheader("Distribuição Temática")
        temas_inst = df_inst_det['nome_topico'].value_counts().head(5).reset_index()
        temas_inst.columns = ['tema', 'count']
        temas_inst['tema_simples'] = temas_inst['tema'].apply(simplificar_topico)
        fig_temas_inst = px.bar(temas_inst, x='tema_simples', y='count', labels={'count': 'TCCs', 'tema_simples': 'Tema'})
        fig_temas_inst.update_layout(height=300, showlegend=False)
        st.plotly_chart(fig_temas_inst, config = {'responsive': True})
