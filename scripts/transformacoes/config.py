# C:\...\transformacoes\config.py

import json
from pathlib import Path

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
