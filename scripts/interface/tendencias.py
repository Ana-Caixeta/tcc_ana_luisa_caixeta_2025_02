# -*- coding: utf-8 -*-
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utilitarios import prever_tendencias, extrair_termos_emergentes, simplificar_topico
from sklearn.linear_model import LinearRegression
import numpy as np

def exibir(df):
    st.subheader("Análise de Tendências e Previsões com Machine Learning")
    st.info("Esta análise utiliza modelos simples para identificar tendências e prever temas em ascensão")

    col_config1, col_config2 = st.columns([3, 1])
    with col_config1:
        st.write("**Configurações de Análise:**")
    with col_config2:
        anos_previsao = st.selectbox("Anos para previsão", [2, 3, 4, 5], index=1)

    if st.button("Executar Análise de Tendências", type="primary"):
        with st.spinner("Processando dados e treinando modelos..."):
            st.markdown("---")
            st.subheader("Tendências por Tema")
            df_tendencias = prever_tendencias(df, anos_previsao=anos_previsao)

            if not df_tendencias.empty:
                df_tendencias['classificacao'] = df_tendencias['score_tendencia'].apply(
                    lambda x: 'Alta Crescimento' if x > 2 else
                              ('Crescimento Moderado' if x > 0 else
                               ('Declínio Moderado' if x > -2 else 'Forte Declínio'))
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

                st.write("**Top 10 Temas com Maior Potencial de Crescimento:**")
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
                st.plotly_chart(fig_tendencias, config = {'responsive': True})

                st.write("**Análise Detalhada de Tendências:**")
                df_display = df_tendencias[['tema_simples', 'ultimo_valor', 'previsao_media', 'percentual_mudanca', 'classificacao']].copy()
                df_display.columns = ['Tema', 'TCCs Atuais', 'Previsão Média', 'Mudança %', 'Tendência']
                df_display = df_display.sort_values('Mudança %', ascending=False)
                df_display['TCCs Atuais'] = df_display['TCCs Atuais'].round(0).astype(int)
                df_display['Previsão Média'] = df_display['Previsão Média'].round(1)
                df_display['Mudança %'] = df_display['Mudança %'].round(1)
                st.dataframe(df_display, hide_index=True, width='stretch', height=400)
            else:
                st.warning("Dados insuficientes para análise de tendências. Ajuste os filtros.")

            st.markdown("---")
            st.subheader("Termos e Conceitos Emergentes")
            df_emergentes = extrair_termos_emergentes(df, top_n=15)
            if not df_emergentes.empty:
                col_left, col_right = st.columns([1, 1])
                with col_left:
                    st.write("**Top 15 Termos em Ascensão:**")
                    df_emergentes_chart = df_emergentes.sort_values('crescimento_pct', ascending=True)
                    fig_emergentes = px.bar(df_emergentes_chart.head(15), x='crescimento_pct', y='termo', orientation='h',
                                            labels={'crescimento_pct': 'Crescimento (%)', 'termo': 'Termo'})
                    fig_emergentes.update_layout(height=500, showlegend=False, yaxis_title="")
                    st.plotly_chart(fig_emergentes, config = {'responsive': True})
                with col_right:
                    st.write("**Detalhamento dos Termos:**")
                    df_emerg_display = df_emergentes[['termo', 'freq_antiga', 'freq_recente', 'crescimento_pct']].copy()
                    df_emerg_display.columns = ['Termo', 'Freq. Antiga', 'Freq. Recente', 'Crescimento %']
                    df_emerg_display['Crescimento %'] = df_emerg_display['Crescimento %'].round(1)
                    st.dataframe(df_emerg_display, hide_index=True, width='stretch', height=500)
            else:
                st.warning("Não foi possível identificar termos emergentes. Verifique os dados.")

            st.markdown("---")
            st.subheader("Previsão de Produção por Tema")
            st.write(f"Previsão da quantidade de TCCs para os próximos {anos_previsao} anos")
            top_5_temas = df['nome_topico'].value_counts().head(5).index.tolist()
            if top_5_temas:
                tema_viz = st.selectbox("Selecione um tema para visualizar a previsão", options=top_5_temas, format_func=simplificar_topico)
                if tema_viz:
                    df_tema_hist = df[df['nome_topico'] == tema_viz].groupby('ano').size().reset_index(name='count')
                    if len(df_tema_hist) >= 2:
                        X_hist = df_tema_hist['ano'].values.reshape(-1, 1)
                        y_hist = df_tema_hist['count'].values
                        model = LinearRegression()
                        model.fit(X_hist, y_hist)
                        ano_max_hist = df['ano'].max()
                        anos_futuro = np.array([ano_max_hist + i for i in range(1, anos_previsao + 1)]).reshape(-1, 1)
                        previsoes = model.predict(anos_futuro)
                        previsoes = np.maximum(previsoes, 0)
                        df_previsao = {'ano': anos_futuro.flatten(), 'count': previsoes, 'tipo': ['Previsão']*len(previsoes)}
                        import pandas as _pd
                        df_previsao = _pd.DataFrame(df_previsao)
                        df_historico = df_tema_hist.copy()
                        df_historico['tipo'] = 'Histórico'
                        df_viz = _pd.concat([df_historico, df_previsao])
                        fig_previsao = px.line(df_viz, x='ano', y='count', color='tipo', markers=True, labels={'count': 'Quantidade de TCCs', 'ano': 'Ano'})
                        fig_previsao.update_layout(height=400)
                        st.plotly_chart(fig_previsao, config = {'responsive': True})
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
        st.info("Clique no botão acima para executar a análise de tendências com Machine Learning")
        st.markdown("---")
        st.subheader("Metodologia")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write("**Regressão Linear**")
            st.write("Análise de tendências temporais para cada tema")
        with col2:
            st.write("**Análise de Crescimento**")
            st.write("Comparação entre períodos para identificar termos emergentes")
        with col3:
            st.write("**Previsão**")
            st.write("Projeção de TCCs para os próximos anos")
