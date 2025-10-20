# -*- coding: utf-8 -*-
"""
Script de Pré-processamento e Modelagem de Tópicos (Versão Completa e Corrigida).

Este script lê os dados já limpos do Data Mart ('datamart.db')
e realiza a modelagem de tópicos, gerando o arquivo final para o dashboard.
"""

import sqlite3
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import time

# --- CONFIGURAÇÕES ---
PROCESSED_DB_NAME = "datamart.db"
OUTPUT_FILENAME = "tccs_dashboard.parquet"
N_TOPICS = 10

# --- FUNÇÕES AUXILIARES ---

def setup_nltk():
    """Baixa as stopwords do NLTK se ainda não estiverem disponíveis."""
    try:
        stopwords.words('portuguese')
    except LookupError:
        print("--> Baixando recursos do NLTK (stopwords)...")
        nltk.download('stopwords')
        print("--> Download concluído.")

def preprocess_text(text):
    """Limpa e normaliza o texto para o modelo."""
    if not isinstance(text, str):
        return ""
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text, re.I|re.A)
    tokens = text.split()
    stop_words = set(stopwords.words('portuguese'))
    tokens = [word for word in tokens if len(word) > 2 and word not in stop_words]
    return " ".join(tokens)

def get_topic_name(topic_idx, top_words):
    """Cria um nome de tópico legível a partir de suas palavras-chave."""
    capitalized_words = [word.capitalize() for word in top_words]
    return f"Tópico {topic_idx}: {', '.join(capitalized_words)}"

def load_data_from_datamart(db_name):
    """Carrega e junta dados do Star Schema para criar uma visão plana."""
    print(f"1. Carregando dados do Data Mart '{db_name}'...")
    try:
        with sqlite3.connect(db_name) as conn:
            query = """
                SELECT
                    t.tcc_id,
                    t.titulo,
                    t.resumo,
                    t.ano,
                    i.sigla as instituicao,
                    c.nome_curso as curso,
                    GROUP_CONCAT(p_aluno.nome_pessoa) as autores,
                    p_orientador.nome_pessoa as orientador
                FROM fato_tcc t
                LEFT JOIN dim_instituicao i ON t.instituicao_id = i.instituicao_id
                LEFT JOIN dim_curso c ON t.curso_id = c.curso_id
                LEFT JOIN ponte_tcc_aluno pta ON t.tcc_id = pta.tcc_id
                LEFT JOIN dim_pessoa p_aluno ON pta.aluno_id = p_aluno.pessoa_id
                LEFT JOIN ponte_tcc_orientador pto ON t.tcc_id = pto.tcc_id
                LEFT JOIN dim_pessoa p_orientador ON pto.orientador_id = p_orientador.pessoa_id
                GROUP BY t.tcc_id
            """
            df = pd.read_sql_query(query, conn)
        print(f"   - {len(df)} registros carregados.")
        return df
    except Exception as e:
        print(f"   - ERRO AO CARREGAR DADOS: {e}")
        return None

# --- FUNÇÃO PRINCIPAL ---

def main():
    """Função principal que orquestra todo o processo."""
    # Adicionamos este print para garantir que o script iniciou
    print("\n--- Iniciando script preprocess.py ---")
    start_time = time.time()
    
    setup_nltk()
    df = load_data_from_datamart(PROCESSED_DB_NAME)

    if df is None or df.empty:
        print("\nProcesso interrompido. Verifique se o script 'etl_star_schema.py' foi executado com sucesso e gerou o 'datamart.db'.")
        return

    print("2. Realizando pré-processamento dos textos...")
    df['texto_completo'] = df['titulo'].fillna('') + ' ' + df['resumo'].fillna('')
    df['resumo_processado'] = df['texto_completo'].apply(preprocess_text)
    
    df.dropna(subset=['resumo_processado'], inplace=True)
    df = df[df['resumo_processado'] != '']
    print(f"   - {len(df)} registros restantes após limpeza.")

    # 3. Modelagem de Tópicos (LDA)
    print(f"3. Executando Modelagem de Tópicos (LDA) para encontrar {N_TOPICS} temas...")
    vectorizer = CountVectorizer(max_df=0.9, min_df=20, max_features=2000, ngram_range=(1,2))
    X = vectorizer.fit_transform(df['resumo_processado'])
    
    lda = LatentDirichletAllocation(n_components=N_TOPICS, random_state=42, n_jobs=-1)
    topic_results = lda.fit_transform(X)
    print("   - Modelagem concluída.")

    # 4. Gerar nomes para os temas
    print("4. Gerando nomes interpretáveis para os temas...")
    feature_names = vectorizer.get_feature_names_out()
    topic_name_mapping = {}
    for topic_idx, topic_component in enumerate(lda.components_):
        top_words = [feature_names[i] for i in topic_component.argsort()[:-4:-1]]
        topic_name = get_topic_name(topic_idx, top_words)
        topic_name_mapping[topic_idx] = topic_name
        print(f"   - {topic_name}")

    # 5. Atribuir temas a cada TCC
    print("5. Atribuindo o tema principal a cada TCC...")
    df['id_topico'] = topic_results.argmax(axis=1)
    df['nome_topico'] = df['id_topico'].map(topic_name_mapping)
    print("   - Atribuição concluída.")

    # 6. Salvar o resultado final
    print(f"6. Salvando o DataFrame enriquecido em '{OUTPUT_FILENAME}'...")
    df_final = df[['titulo', 'autores', 'ano', 'instituicao', 'resumo', 'resumo_processado', 'curso', 'nome_topico', 'orientador']]
    df_final.to_parquet(OUTPUT_FILENAME, index=False)
    print("   - Arquivo salvo com sucesso!")
    
    end_time = time.time()
    print(f"\n--- Processo finalizado em {end_time - start_time:.2f} segundos. ---")


if __name__ == "__main__":
    main()
