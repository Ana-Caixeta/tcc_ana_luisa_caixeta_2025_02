# -*- coding: utf-8 -*-
import streamlit as st
import plotly.express as px
from utilitarios import simplificar_topico, metric_bold

def exibir(df):
    st.subheader("Visão Geral")
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_bold("Total de TCCs", f"{len(df):,}".replace(",", "."))
    with col2:
        metric_bold("Instituições", df['instituicao'].nunique())
    with col3:
        metric_bold("Orientadores", df['orientador'].nunique())
    with col4:
        metric_bold("Temas", df['nome_topico'].nunique())

    st.markdown("---")

    # Produção anual
    col_left, col_right = st.columns(2)
    with col_left:
        st.subheader("Produção Anual de TCCs")
        df_ano = df.groupby('ano').size().reset_index(name='count')
        fig_ano = px.bar(df_ano, x='ano', y='count', labels={'count': 'Quantidade', 'ano': 'Ano'})
        fig_ano.update_layout(height=400, showlegend=False, yaxis_title="Quantidade de TCCs")
        st.plotly_chart(fig_ano, config = {'responsive': True})

    with col_right:
        st.subheader("Distribuição por Tema")
        df_topicos = df['nome_topico'].value_counts().head(8).reset_index()
        df_topicos.columns = ['tema', 'count']
        df_topicos['tema_simples'] = df_topicos['tema'].apply(simplificar_topico)
        fig_pizza = px.pie(df_topicos, values='count', names='tema_simples', hole=0.4)
        fig_pizza.update_layout(height=400, showlegend=True)
        st.plotly_chart(fig_pizza, config = {'responsive': True})

    st.markdown("---")

    # Top instituições e cursos
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Top 5 Instituições")
        top_inst = df['instituicao'].value_counts().head(5).reset_index()
        top_inst.columns = ['Instituição', 'TCCs']
        st.dataframe(top_inst, hide_index=True, width='stretch')
    with col2:
        st.subheader("Top 5 Cursos")
        top_cursos = df['curso'].value_counts().head(5).reset_index()
        top_cursos.columns = ['Curso', 'TCCs']
        st.dataframe(top_cursos, hide_index=True, width='stretch')


    st.markdown("---")
    st.subheader("Tabela Completa de Dados")
    st.caption("Visualização de todos os TCCs conforme os filtros aplicados.")
    total_linhas = len(df)
    st.markdown(f"**Total exibido:** {total_linhas:,} registros".replace(",", "."))

    df_ordenado = df.sort_values(by='ano', ascending=False)
    st.dataframe(df_ordenado, width="stretch", hide_index=True)