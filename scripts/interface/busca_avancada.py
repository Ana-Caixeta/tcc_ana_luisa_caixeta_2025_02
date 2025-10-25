# -*- coding: utf-8 -*-
import streamlit as st
import plotly.express as px
from utilitarios import calcular_similaridade, simplificar_topico

def exibir(df):
    st.subheader("Busca Avançada e Similaridade")
    col1, col2 = st.columns([3, 1])
    with col1:
        busca = st.text_input("Buscar em títulos e resumos", "")
    with col2:
        limite_resultados = st.number_input("Limite", min_value=5, max_value=100, value=20, step=5)

    if busca:
        mask = (
            df['titulo'].str.contains(busca, case=False, na=False) |
            df['resumo'].str.contains(busca, case=False, na=False)
        )
        df_busca = df[mask].head(limite_resultados)
        st.success(f"Encontrados {len(df_busca)} resultados")
        for idx, row in df_busca.iterrows():
            with st.expander(f"{row['titulo']}"):
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

                if st.button(f"Encontrar TCCs similares", key=f"sim_{idx}"):
                    st.session_state[f'buscar_similar_{idx}'] = True

                if st.session_state.get(f'buscar_similar_{idx}', False):
                    with st.spinner("Calculando similaridade..."):
                        df_reset = df.reset_index(drop=True)
                        idx_matches = df_reset[df_reset['titulo'] == row['titulo']].index
                        if not idx_matches.empty:
                            idx_rel = idx_matches[0]
                            df_similar = calcular_similaridade(df_reset, idx_rel, top_n=5)
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
        st.info("Digite um termo para buscar em títulos e resumos dos TCCs")

    st.markdown("---")
    st.subheader("Análise de Similaridade entre TCCs")
    col1, col2 = st.columns([3, 1])
    with col1:
        titulos_disponiveis = df['titulo'].tolist()
        tcc_selecionado = st.selectbox("Selecione um TCC para encontrar trabalhos similares", options=titulos_disponiveis, key="similarity_selector")
    with col2:
        num_similares = st.number_input("Quantidade", min_value=3, max_value=20, value=5, step=1, key="num_sim")

    if tcc_selecionado and st.button("Buscar TCCs Similares", key="btn_similarity"):
        with st.spinner("Analisando similaridade..."):
            df_reset = df.reset_index(drop=True)
            idx_sel = df_reset[df_reset['titulo'] == tcc_selecionado].index[0]
            tcc_info = df_reset.iloc[idx_sel]
            col_a, col_b = st.columns(2)
            with col_a:
                st.write("**TCC de Referência:**")
                st.write(f"**Título:** {tcc_info['titulo']}")
                st.write(f"**Autores:** {tcc_info['autores']}")
                st.write(f"**Ano:** {tcc_info['ano']}")
            with col_b:
                st.write(f"**Instituição:** {tcc_info['instituicao']}")
                st.write(f"**Curso:** {tcc_info['curso']}")
                st.write(f"**Tema:** {simplificar_topico(tcc_info['nome_topico'])}")

            df_similar = calcular_similaridade(df_reset, idx_sel, top_n=num_similares)
            if not df_similar.empty:
                st.markdown("---")
                st.write(f"**Top {num_similares} TCCs Mais Similares:**")
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
                st.write("**Visualização de Similaridade:**")
                df_sim_viz = df_similar.copy().sort_values('similaridade', ascending=True)
                df_sim_viz['titulo_curto'] = df_sim_viz['titulo'].apply(lambda x: x[:40] + "..." if len(x) > 40 else x)
                df_sim_viz['similaridade_pct'] = df_sim_viz['similaridade'] * 100
                fig_sim = px.bar(df_sim_viz, x='similaridade_pct', y='titulo_curto', orientation='h', labels={'similaridade_pct': 'Similaridade (%)', 'titulo_curto': 'TCC'})
                fig_sim.update_layout(height=400, showlegend=False, yaxis_title="")
                st.plotly_chart(fig_sim, config = {'responsive': True})
            else:
                st.warning("Não foi possível calcular similaridades.")
