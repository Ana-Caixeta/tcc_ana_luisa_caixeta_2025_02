# -*- coding: utf-8 -*-
"""
Script ETL para Extrair, Transformar e Carregar dados de TCCs
(Versão com Validação de Instituição baseada em Query SQL)
"""

import sqlite3
import pandas as pd
import re
import time
from sqlalchemy import create_engine
import unicodedata
from config import carregar_instituicoes
import os

# Configuração
RAW_DB_NAME = "integra.db"
PROCESSED_DB_NAME = "datamart.db"
PROCESSED_DB_ENGINE = f"sqlite:///{PROCESSED_DB_NAME}"

# Arquivo para logar TCCs que foram descartados por falhas no mapeamento
LOG_REJEITADOS_FILE = "log_tccs_rejeitados.csv"

# Carregar instituições
print("Carregando dicionário de instituições...")
INSTITUICOES = carregar_instituicoes()

#Funções Auxiliares
def normalize_string(text):
    """
    Remove acentos, converte para minúsculas, remove caracteres não-ASCII
    e corrige erros de digitação comuns de 'instituto'.
    """
    if not isinstance(text, str): return ""
    text = unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode('utf-8')
    text = text.lower()
    
    # Corrige erros de digitação comuns para "instituto"
    text = text.replace("instituicao", "instituto") # Handle 'instituição'
    text = text.replace("institituto", "instituto") # Handle 'institituto'
    text = text.replace("instituo", "instituto")    # Handle 'instituo'
    
    return text

def init_cap(series): 
    """Converte uma Série pandas para Title Case (primeira letra maiúscula)."""
    return series.astype(str).str.title().str.strip()

def extrair_autores_orientador(autores_str):
    """Separa a string de autores em uma lista de alunos e um orientador."""
    if not isinstance(autores_str, str): return [], None
    alunos, orientador = [], None
    partes = [p.strip() for p in autores_str.split(',')]
    for parte in partes:
        if "(Orientador/a)" in parte: 
            orientador = parte.replace("(Orientador/a)", "").strip()
        else: 
            alunos.append(parte)
    return alunos, orientador

def validar_tcc_rede_federal(row):
    """
    Valida se o TCC pertence à rede federal, com base na lógica das queries SQL.
    
    Um TCC encontrado no site 'sigla_alvo_coleta' (ex: "IFB") é válido se
    o seu nome 'nome_tcc_bruto' (ex: "Institituto Federal de Brasília")
    contiver "instituto federal" (corrigido) OU a 'sigla_alvo' (normalizada).
    
    Retorna a sigla-alvo se for válido, ou None se for de outra instituição.
    """
    sigla_alvo = row['sigla_alvo_coleta']
    nome_bruto_tcc = row['nome_tcc_bruto']
    
    # Se não temos a sigla-alvo ou o nome bruto, não podemos validar
    if pd.isna(sigla_alvo) or pd.isna(nome_bruto_tcc):
        return None
        
    norm_text = normalize_string(nome_bruto_tcc)
    norm_sigla = normalize_string(sigla_alvo)

    # Check 1: (like '%INSTITUTO FEDERAL%') com typos corrigidos
    is_general_federal = "instituto federal" in norm_text
    
    # Check 2: (like '%IFB%')
    # Usamos f" {norm_sigla} " para evitar falsos positivos (ex: "PROJETO IF" não bateria "IF")
    # Mas como o usuário usou LIKE '%IFB%', vamos manter a lógica de 'contém'
    is_specific_sigla = norm_sigla in norm_text
    
    if is_general_federal or is_specific_sigla:
        return sigla_alvo # SUCESSO! (Ex: "IFB" ou "Instituto Federal...")
    
    # FALHA: (ex: "Universidade de Brasilia" não contém "instituto federal" nem "ifb")
    return None

def logar_rejeitados(df_rejeitados, motivo_rejeicao, arquivo_log, modo='w'):
    """Salva os TCCs rejeitados em um CSV para análise posterior."""
    if df_rejeitados.empty:
        return
        
    print(f"     - AVISO: {len(df_rejeitados)} registros serão descartados por '{motivo_rejeicao}'.")
    print(f"     - Logando rejeitados em '{arquivo_log}'...")
    
    df_log = df_rejeitados.copy()
    df_log['motivo_rejeicao'] = motivo_rejeicao
    
    # Determina se escreve o cabeçalho
    escrever_cabecalho = (modo == 'w') or (not os.path.exists(arquivo_log))
    
    df_log.to_csv(arquivo_log, mode=modo, index=False, header=escrever_cabecalho, encoding='utf-8-sig')

# --- Função Principal do ETL ---

def main():
    start_time = time.time()
    print("Iniciando processo ETL para o Star Schema (Validação por Query)")
    
    # Limpa o arquivo de log antigo, se existir
    if os.path.exists(LOG_REJEITADOS_FILE):
        os.remove(LOG_REJEITADOS_FILE)

    # 0. Preparar Mapa de Nomes (para a Dimensão)
    map_nomes_completos = {sigla: valores[0] for sigla, valores in INSTITUICOES.items()}

    # 1. Extrair dados
    print(f"\n1. Extraindo dados de '{RAW_DB_NAME}'...")
    try:
        with sqlite3.connect(RAW_DB_NAME) as conn:
            # Assume que a coluna de sigla no BD bruto se chama 'sigla'
            # e a coluna de nome bruto se chama 'instituicao'
            query = "SELECT *, instituicao as nome_tcc_bruto, sigla as sigla_alvo_coleta FROM tccs"
            df_raw = pd.read_sql_query(query, conn)
        print(f"   - Total de registros brutos extraídos: {len(df_raw)}")
    except Exception as e:
        print(f"   - ERRO: Falha ao ler o banco de dados bruto '{RAW_DB_NAME}'.")
        print(f"   - Detalhe: {e}")
        print("   - Verifique se a coluna 'sigla' existe na tabela 'tccs'.")
        return

    # 2. Transformar os dados
    print("\n2. Transformando dados...")

    df = df_raw.copy()
    print(f"   - Registros para processar (sem filtro de nível): {len(df)}")
    
    # --- Criação da Dimensão Instituição ---
    print("\n   - Criando Dimensão Instituição...")
    df_instituicao = pd.DataFrame.from_dict(INSTITUICOES, orient='index', columns=['nome_completo', 'url', 'uf'])
    df_instituicao['sigla'] = df_instituicao.index
    df_instituicao['nome_completo'] = df_instituicao['sigla'].map(map_nomes_completos)
    df_instituicao.reset_index(drop=True, inplace=True)
    df_instituicao['instituicao_id'] = df_instituicao.index + 1
    dim_instituicao = df_instituicao[['instituicao_id', 'sigla', 'nome_completo', 'uf', 'url']]
    
    # --- Validação da Instituição (Lógica Central) ---
    print("   - Validando TCCs da Rede Federal (lógica SQL)...")
    df['sigla_mapeada'] = df.apply(validar_tcc_rede_federal, axis=1)
    
    # Log de Rejeitados 1: Instituição Inválida
    mapeados_com_sucesso = df['sigla_mapeada'].notna().sum()
    df_rejeitados_inst = df[df['sigla_mapeada'].isna()]
    logar_rejeitados(df_rejeitados_inst, "Instituição do TCC não parece ser da Rede Federal (ex: Universidade)", LOG_REJEITADOS_FILE, modo='w')
    print(f"     - {mapeados_com_sucesso} de {len(df)} registros foram VALIDADOS como pertencentes à Rede Federal.")
    
    df.dropna(subset=['sigla_mapeada'], inplace=True)
    if len(df) == 0: 
        print("   - Nenhum registro validado. Encerrando.")
        return

    # --- Continuação da Transformação ---
    df['lista_alunos'] = df['autores'].apply(extrair_autores_orientador).apply(lambda x: x[0])
    df['orientador'] = df['autores'].apply(extrair_autores_orientador).apply(lambda x: x[1])
    
    # --- Criação das outras Dimensões ---
    print("   - Criando Dimensões Campus, Curso e Pessoa...")
    dim_campus = pd.DataFrame(df['campus'].dropna().unique(), columns=['nome_campus']); dim_campus['nome_campus'] = init_cap(dim_campus['nome_campus']); dim_campus.sort_values('nome_campus', inplace=True); dim_campus['campus_id'] = range(1, len(dim_campus) + 1)
    dim_curso = pd.DataFrame(df['curso'].dropna().unique(), columns=['nome_curso']); dim_curso['nome_curso'] = init_cap(dim_curso['nome_curso']); dim_curso['nivel'] = 'N/A'; dim_curso.sort_values('nome_curso', inplace=True); dim_curso['curso_id'] = range(1, len(dim_curso) + 1)
    pessoas_unicas = pd.concat([df['lista_alunos'].explode(), df['orientador']]).dropna().unique(); dim_pessoa = pd.DataFrame(pessoas_unicas, columns=['nome_pessoa']); dim_pessoa['nome_pessoa'] = init_cap(dim_pessoa['nome_pessoa']); dim_pessoa.sort_values('nome_pessoa', inplace=True); dim_pessoa['pessoa_id'] = range(1, len(dim_pessoa) + 1)
    
    print("\n   - Criando Tabela Fato e Pontes...")
    df['tcc_id'] = range(1, len(df) + 1)
    
    # Criar mapas de chaves (FKs)
    map_instituicao = pd.Series(dim_instituicao.instituicao_id.values, index=dim_instituicao.sigla).to_dict()
    map_campus = pd.Series(dim_campus.campus_id.values, index=dim_campus.nome_campus).to_dict()
    map_curso = pd.Series(dim_curso.curso_id.values, index=dim_curso.nome_curso).to_dict()
    map_pessoa = pd.Series(dim_pessoa.pessoa_id.values, index=dim_pessoa.nome_pessoa).to_dict()

    # Mapear chaves estrangeiras
    df['instituicao_id'] = df['sigla_mapeada'].map(map_instituicao)
    df['campus_id'] = init_cap(df['campus']).map(map_campus)
    df['curso_id'] = init_cap(df['curso']).map(map_curso)
    
    # --- Filtro 2: Garantia de Integridade das Dimensões ---
    print("\n   - ETAPA DE FILTRO 2: Garantia de Integridade das Dimensões (Campus/Curso)")
    colunas_fk = ['instituicao_id', 'campus_id', 'curso_id']
    
    # Log de Rejeitados 2: Campus/Curso Nulos (ex: nome de campus não mapeado)
    df_rejeitados_fk = df[df[colunas_fk].isna().any(axis=1)]
    logar_rejeitados(df_rejeitados_fk, "Falha ao mapear FK (Campus ou Curso nulo/inválido)", LOG_REJEITADOS_FILE, modo='a')

    df.dropna(subset=colunas_fk, inplace=True)
    print(f"     - Registros restantes após garantir mapeamento FK: {len(df)}")
    
    # --- Montagem Final das Tabelas ---
    fato_tcc = df[['tcc_id', 'titulo', 'resumo', 'palavras_chaves', 'ano', 'curso_id', 'instituicao_id', 'campus_id']]
    
    ponte_tcc_aluno = df[['tcc_id', 'lista_alunos']].explode('lista_alunos').rename(columns={'lista_alunos': 'nome_pessoa'})
    ponte_tcc_aluno['aluno_id'] = init_cap(ponte_tcc_aluno['nome_pessoa']).map(map_pessoa)
    ponte_tcc_aluno = ponte_tcc_aluno[['tcc_id', 'aluno_id']].dropna()
    
    ponte_tcc_orientador = df[['tcc_id', 'orientador']].rename(columns={'orientador': 'nome_pessoa'})
    ponte_tcc_orientador['orientador_id'] = init_cap(ponte_tcc_orientador['nome_pessoa']).map(map_pessoa)
    ponte_tcc_orientador = ponte_tcc_orientador[['tcc_id', 'orientador_id']].dropna()

    # 3. Carregar dados no Data Mart
    print(f"\n3. Carregando {len(fato_tcc)} registros no Data Mart '{PROCESSED_DB_NAME}'...")
    engine = create_engine(PROCESSED_DB_ENGINE)
    
    try:
        dim_instituicao.to_sql('dim_instituicao', engine, if_exists='replace', index=False)
        dim_campus.to_sql('dim_campus', engine, if_exists='replace', index=False)
        dim_curso.to_sql('dim_curso', engine, if_exists='replace', index=False)
        dim_pessoa.to_sql('dim_pessoa', engine, if_exists='replace', index=False)
        fato_tcc.to_sql('fato_tcc', engine, if_exists='replace', index=False)
        ponte_tcc_aluno.to_sql('ponte_tcc_aluno', engine, if_exists='replace', index=False)
        ponte_tcc_orientador.to_sql('ponte_tcc_orientador', engine, if_exists='replace', index=False)
        
        print("   - Carga de dados concluída.")
    
    except Exception as e:
        print(f"   - ERRO: Falha ao carregar dados no Data Mart.")
        print(f"   - Detalhe: {e}")
        return

    end_time = time.time()
    print(f"\n--- Processo ETL finalizado em {end_time - start_time:.2f} segundos. ---")

if __name__ == "__main__":
    main()
