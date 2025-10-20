# -*- coding: utf-8 -*-
import streamlit as st
import plotly.express as px
from utilitarios import simplificar_topico, metric_bold

def exibir(df):
    st.subheader("Análise de Orientadores")
    df_orient = df.groupby('orientador').agg({
        'titulo': 'count',
        'nome_topico': lambda x: x.mode()[0] if not x.mode().empty else 'N/A'
    }).reset_index()
    df_orient.columns = ['orientador', 'qtd_orientacoes', 'tema_principal']
    df_orient = df_orient.sort_values('qtd_orientacoes', ascending=False)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_bold("Total de Orientadores", len(df_orient))
    with col2:
        metric_bold("Média Orientações Por Professor", f"{df_orient['qtd_orientacoes'].mean():.1f}")
    with col3:
        max_orientacoes = int(df_orient['qtd_orientacoes'].max()) if not df_orient.empty else 0
        metric_bold("Máximo Orientações", max_orientacoes)
    with col4:
        orientador_top = df_orient.iloc[0]['orientador'] if not df_orient.empty else 'N/A'
        metric_bold("Orientador Mais Ativo", orientador_top)

    st.markdown("---")

    st.subheader("Top 15 Orientadores")
    top_orient = df_orient.head(15).copy().sort_values('qtd_orientacoes', ascending=True)
    fig_orient = px.bar(top_orient, x='qtd_orientacoes', y='orientador', orientation='h', labels={'qtd_orientacoes': 'Orientações', 'orientador': 'Orientador'})
    fig_orient.update_layout(height=600, showlegend=False, yaxis_title="")
    st.plotly_chart(fig_orient, use_container_width=True)

    st.markdown("---")
    
    st.subheader("Detalhes por Orientador")
    orientador_selecionado = st.selectbox("Escolha um orientador", options=df_orient['orientador'].tolist())
    if orientador_selecionado:
        df_prof = df[df['orientador'] == orientador_selecionado]
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
        fig_prof_tempo = px.line(df_prof_tempo, x='ano', y='count', markers=True, labels={'count': 'Orientações', 'ano': 'Ano'})
        fig_prof_tempo.update_layout(height=250, showlegend=False)
        st.plotly_chart(fig_prof_tempo, use_container_width=True)

    st.markdown("---")
    st.subheader("Ranking Completo de Orientadores")
    df_orient_display = df_orient.copy()
    df_orient_display['tema_simples'] = df_orient_display['tema_principal'].apply(simplificar_topico)
    df_orient_display = df_orient_display[['orientador', 'qtd_orientacoes', 'tema_simples']]
    df_orient_display.columns = ['Orientador', 'Orientações', 'Tema Principal']
    st.dataframe(df_orient_display, hide_index=True, use_container_width=True, height=400)
