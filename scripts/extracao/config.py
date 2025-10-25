# C:\...\extracao\config.py

import json
from pathlib import Path

# CONFIGURAÇÕES GERAIS
DB_NAME = "integra.db"
PAGE_SIZE = 50
MAX_CONCURRENT = 50

# HEADERS HTTP
DEFAULT_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/116.0.0.0 Safari/537.36",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
}

# INSTITUIÇÕES
def carregar_instituicoes():
    """Carrega a lista de instituições de um arquivo JSON na mesma pasta."""
    
    # Constrói um caminho absoluto para o arquivo JSON, garantindo que ele seja encontrado
    caminho = Path(__file__).parent.parent / "lista_instituicoes.json"
    try:
        with open(caminho, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Erro: Arquivo de instituições não encontrado em '{caminho}'")
        return {}

INSTITUICOES = carregar_instituicoes()
