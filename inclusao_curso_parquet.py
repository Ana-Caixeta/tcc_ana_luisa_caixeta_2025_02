import pandas as pd
import json
import os

# CONFIGURAÇÕES DE CAMINHOS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Caminho do arquivo JSON
JSON_PATH = os.path.join(BASE_DIR, "agrupamentos_cursos.json")

# Caminho do arquivo Parquet (ajuste se a pasta scripts estiver em outro lugar)
PARQUET_PATH = os.path.join(BASE_DIR, "scripts", "interface", "tccs_dashboard.parquet")

def carregar_regras_json(caminho_json):
    """Lê o arquivo JSON e cria o dicionário de mapeamento invertido."""
    if not os.path.exists(caminho_json):
        raise FileNotFoundError(f"Arquivo JSON não encontrado em: {caminho_json}")

    print(f"1. Lendo regras de: {caminho_json}...")
    with open(caminho_json, 'r', encoding='utf-8') as f:
        dados_json = json.load(f)

    # Inverter a lógica para o Pandas (De -> Para)
    # O JSON é: { "Nome Oficial": ["Variação 1", "Variação 2"] }
    # O Pandas precisa: { "Variação 1": "Nome Oficial", "Variação 2": "Nome Oficial" }
    mapa_invertido = {}
    
    for nome_unificado, lista_variacoes in dados_json.items():
        for variacao in lista_variacoes:
            mapa_invertido[variacao] = nome_unificado
            
    return mapa_invertido

def processar_parquet():
    # 1. Carregar as regras do JSON
    try:
        mapa_de_para = carregar_regras_json(JSON_PATH)
    except Exception as e:
        print(f"ERRO CRÍTICO: {e}")
        return

    # 2. Verificar e Carregar o Parquet
    if not os.path.exists(PARQUET_PATH):
        print(f"ERRO: Arquivo Parquet não encontrado em: {PARQUET_PATH}")
        return

    print(f"2. Lendo Parquet: {PARQUET_PATH}")
    try:
        df = pd.read_parquet(PARQUET_PATH)
    except Exception as e:
        print(f"ERRO ao ler parquet: {e}")
        return

    # 3. Aplicar a unificação
    print("3. Criando coluna 'curso_unificado'...")
    
    # Mapeia usando o dicionário criado a partir do JSON
    serie_mapeada = df['curso'].map(mapa_de_para)
    
    # Preenche quem não foi encontrado com o valor original
    df['curso_unificado'] = serie_mapeada.fillna(df['curso'])

    # Feedback visual
    total_linhas = len(df)
    linhas_alteradas = len(df[df['curso'] != df['curso_unificado']])
    print(f"   - Total de registros processados: {total_linhas}")
    print(f"   - Registros unificados/alterados: {linhas_alteradas}")

    # 4. Salvar o resultado
    print(f"4. Salvando alterações...")
    df.to_parquet(PARQUET_PATH, index=False)
    print("--- Processo Finalizado com Sucesso! ---")

if __name__ == "__main__":
    processar_parquet()
    