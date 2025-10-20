# -*- coding: utf-8 -*-
"""
Script ETL para Extrair, Transformar e Carregar dados de TCCs
(Versão Corrigida para erro de colunas).
"""

import sqlite3
import pandas as pd
import re
import time
from sqlalchemy import create_engine
import unicodedata
from config import carregar_instituicoes

# --- CONFIGURAÇÕES ---
RAW_DB_NAME = "integra.db"
PROCESSED_DB_NAME = "datamart.db"
PROCESSED_DB_ENGINE = f"sqlite:///{PROCESSED_DB_NAME}"

# Buscar dicionário de Instituições do seu arquivo de configuração
INSTITUICOES = carregar_instituicoes()

# Dicionário auxiliar para adicionar os nomes completos que estão faltando
NOMES_COMPLETOS = {
    "IFAC": "Instituto Federal do Acre", "IFAL": "Instituto Federal de Alagoas", "IFAP": "Instituto Federal do Amapá",
    "IFAM": "Instituto Federal do Amazonas", "IFBA": "Instituto Federal da Bahia", "IFB": "Instituto Federal de Brasília",
    "IFCE": "Instituto Federal do Ceará", "IFES": "Instituto Federal do Espírito Santo", "IFG": "Instituto Federal de Goiás",
    "IFGOIANO": "Instituto Federal Goiano", "IFMA": "Instituto Federal do Maranhão", "IFMG": "Instituto Federal de Minas Gerais",
    "IFNMG": "Instituto Federal do Norte de Minas Gerais", "IFSUDESTEMG": "Instituto Federal do Sudeste de Minas Gerais",
    "IFSULDEMINAS": "Instituto Federal do Sul de Minas", "IFTM": "Instituto Federal do Triângulo Mineiro",
    "IFMT": "Instituto Federal do Mato Grosso", "IFMS": "Instituto Federal do Mato Grosso do Sul",
    "IFPA": "Instituto Federal do Pará", "IFPB": "Instituto Federal da Paraíba", "IFPE": "Instituto Federal de Pernambuco",
    "IFSERTAOPE": "Instituto Federal do Sertão Pernambucano", "IFPI": "Instituto Federal do Piauí",
    "IFPR": "Instituto Federal do Paraná", "IFRJ": "Instituto Federal do Rio de Janeiro",
    "IFFLUMINENSE": "Instituto Federal Fluminense", "IFRN": "Instituto Federal do Rio Grande do Norte",
    "IFRO": "Instituto Federal de Rondônia", "IFRR": "Instituto Federal de Roraima",
    "IFRS": "Instituto Federal do Rio Grande do Sul", "IFFARROUPILHA": "Instituto Federal Farroupilha",
    "IFSUL": "Instituto Federal Sul-rio-grandense", "IFSC": "Instituto Federal de Santa Catarina",
    "IFC": "Instituto Federal Catarinense", "IFSP": "Instituto Federal de São Paulo", "IFS": "Instituto Federal de Sergipe",
    "IFTO": "Instituto Federal do Tocantins", "CEFET-RJ": "Centro Federal de Educação Tecnológica Celso Suckow da Fonseca",
    "CEFET-MG": "Centro Federal de Educação Tecnológica de Minas Gerais"
}


# --- FUNÇÕES AUXILIARES ---
def normalize_string(text):
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    return text.lower()

def init_cap(series): return series.astype(str).str.title().str.strip()
def extrair_autores_orientador(autores_str):
    if not isinstance(autores_str, str): return [], None
    alunos, orientador = [], None
    partes = [p.strip() for p in autores_str.split(',')]
    for parte in partes:
        if "(Orientador/a)" in parte: orientador = parte.replace("(Orientador/a)", "").strip()
        else: alunos.append(parte)
    return alunos, orientador

def mapear_sigla_por_nome_completo(texto_instituicao, mapa_nomes):
    texto_normalizado = normalize_string(texto_instituicao)
    for nome_normalizado, sigla in mapa_nomes.items():
        if nome_normalizado in texto_normalizado:
            return sigla
    return None

def main():
    start_time = time.time()
    print("--- Iniciando processo ETL para o Star Schema (Versão Corrigida) ---")

    # 1. EXTRACT
    print(f"\n1. Extraindo dados de '{RAW_DB_NAME}'...")
    with sqlite3.connect(RAW_DB_NAME) as conn:
        df_raw = pd.read_sql_query("SELECT * FROM tccs", conn)
    print(f"   - Total de registros brutos extraídos: {len(df_raw)}")

    # 2. TRANSFORM
    print("\n2. Transformando dados...")
    palavras_chave_superior = ['Bacharelado', 'Tecnologia', 'Licenciatura', 'Engenharia', 'Superior']
    regex_superior = '|'.join(palavras_chave_superior)
    df = df_raw[df_raw['curso'].str.contains(regex_superior, case=False, na=False)].copy()
    print(f"   - Registros restantes após filtro 'Nível Superior': {len(df)}")
    if len(df) == 0: return
    
    # --- Criação da Dimensão Instituição (CORRIGIDO) ---
    print("\n   - Criando Dimensões...")
    # 1. Criar DataFrame com as 2 colunas do seu config.py (url, uf)
    # Assumindo a ordem [url, uf] com base no seu JSON original. Se for [uf, url], inverta aqui.
    df_instituicao = pd.DataFrame.from_dict(INSTITUICOES, orient='index', columns=['url', 'uf'])
    df_instituicao['sigla'] = df_instituicao.index
    # 2. Adicionar a coluna 'nome_completo' a partir do nosso dicionário auxiliar
    df_instituicao['nome_completo'] = df_instituicao['sigla'].map(NOMES_COMPLETOS)
    df_instituicao.reset_index(drop=True, inplace=True); df_instituicao['instituicao_id'] = df_instituicao.index + 1
    dim_instituicao = df_instituicao[['instituicao_id', 'sigla', 'nome_completo', 'uf', 'url']] # Reordenar para o schema
    
    # --- Mapeamento de Siglas (usando a dimensão que acabamos de criar) ---
    # Criar o mapa de busca aqui dentro, com os dados já tratados
    mapa_nomes_normalizados = {
        normalize_string(v['nome_completo']): v['sigla'] 
        for i, v in dim_instituicao.iterrows()
    }
    # Ordenar pelas chaves mais longas
    mapa_nomes_normalizados = dict(sorted(mapa_nomes_normalizados.items(), key=lambda item: len(item[0]), reverse=True))

    print("   - Mapeando siglas de instituição a partir do nome completo...")
    df['sigla_mapeada'] = df['instituicao'].apply(lambda x: mapear_sigla_por_nome_completo(x, mapa_nomes_normalizados))
    
    mapeados_com_sucesso = df['sigla_mapeada'].notna().sum()
    print(f"     - {mapeados_com_sucesso} de {len(df)} registros tiveram uma sigla mapeada com sucesso.")
    df.dropna(subset=['sigla_mapeada'], inplace=True)

    df['lista_alunos'] = df['autores'].apply(extrair_autores_orientador).apply(lambda x: x[0])
    df['orientador'] = df['autores'].apply(extrair_autores_orientador).apply(lambda x: x[1])
    
    # --- Criação das outras Dimensões ---
    dim_campus = pd.DataFrame(df['campus'].dropna().unique(), columns=['nome_campus']); dim_campus['nome_campus'] = init_cap(dim_campus['nome_campus']); dim_campus.sort_values('nome_campus', inplace=True); dim_campus['campus_id'] = range(1, len(dim_campus) + 1)
    dim_curso = pd.DataFrame(df['curso'].dropna().unique(), columns=['nome_curso']); dim_curso['nome_curso'] = init_cap(dim_curso['nome_curso']); dim_curso['nivel'] = 'Superior'; dim_curso.sort_values('nome_curso', inplace=True); dim_curso['curso_id'] = range(1, len(dim_curso) + 1)
    pessoas_unicas = pd.concat([df['lista_alunos'].explode(), df['orientador']]).dropna().unique(); dim_pessoa = pd.DataFrame(pessoas_unicas, columns=['nome_pessoa']); dim_pessoa['nome_pessoa'] = init_cap(dim_pessoa['nome_pessoa']); dim_pessoa.sort_values('nome_pessoa', inplace=True); dim_pessoa['pessoa_id'] = range(1, len(dim_pessoa) + 1)
    
    print("\n   - Criando Tabela Fato e Pontes...")
    df['tcc_id'] = range(1, len(df) + 1)
    
    map_instituicao = pd.Series(dim_instituicao.instituicao_id.values, index=dim_instituicao.sigla).to_dict()
    map_campus = pd.Series(dim_campus.campus_id.values, index=dim_campus.nome_campus).to_dict()
    map_curso = pd.Series(dim_curso.curso_id.values, index=dim_curso.nome_curso).to_dict()

    df['instituicao_id'] = df['sigla_mapeada'].map(map_instituicao)
    df['campus_id'] = init_cap(df['campus']).map(map_campus)
    df['curso_id'] = init_cap(df['curso']).map(map_curso)
    
    print("\n   - ETAPA DE FILTRO 2: Garantia de Integridade das Dimensões")
    df.dropna(subset=['instituicao_id', 'campus_id', 'curso_id'], inplace=True)
    print(f"     - Registros restantes após garantir mapeamento: {len(df)}")
    
    fato_tcc = df[['tcc_id', 'titulo', 'resumo', 'palavras_chaves', 'ano', 'curso_id', 'instituicao_id', 'campus_id']]
    
    map_pessoa = pd.Series(dim_pessoa.pessoa_id.values, index=dim_pessoa.nome_pessoa).to_dict()
    ponte_tcc_aluno = df[['tcc_id', 'lista_alunos']].explode('lista_alunos').rename(columns={'lista_alunos': 'nome_pessoa'}); ponte_tcc_aluno['aluno_id'] = init_cap(ponte_tcc_aluno['nome_pessoa']).map(map_pessoa); ponte_tcc_aluno = ponte_tcc_aluno[['tcc_id', 'aluno_id']].dropna()
    ponte_tcc_orientador = df[['tcc_id', 'orientador']].rename(columns={'orientador': 'nome_pessoa'}); ponte_tcc_orientador['orientador_id'] = init_cap(ponte_tcc_orientador['nome_pessoa']).map(map_pessoa); ponte_tcc_orientador = ponte_tcc_orientador[['tcc_id', 'orientador_id']].dropna()

    print(f"\n3. Carregando {len(fato_tcc)} registros no Data Mart '{PROCESSED_DB_NAME}'...")
    engine = create_engine(PROCESSED_DB_ENGINE)
    dim_instituicao.to_sql('dim_instituicao', engine, if_exists='replace', index=False); dim_campus.to_sql('dim_campus', engine, if_exists='replace', index=False); dim_curso.to_sql('dim_curso', engine, if_exists='replace', index=False); dim_pessoa.to_sql('dim_pessoa', engine, if_exists='replace', index=False); fato_tcc.to_sql('fato_tcc', engine, if_exists='replace', index=False); ponte_tcc_aluno.to_sql('ponte_tcc_aluno', engine, if_exists='replace', index=False); ponte_tcc_orientador.to_sql('ponte_tcc_orientador', engine, if_exists='replace', index=False)
    print("   - Carga de dados concluída.")
    
    end_time = time.time()
    print(f"\n--- Processo ETL finalizado em {end_time - start_time:.2f} segundos. ---")

if __name__ == "__main__":
    main()