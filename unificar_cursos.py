import os
import pandas as pd
import json
import unicodedata
from rapidfuzz import fuzz, process  # mais rápido e leve que fuzzywuzzy

# Antes de rodar dar um pip install pandas pyarrow rapidfuzz


# Caminho do arquivo Parquet
BASE_DIR = os.path.dirname(__file__)
PARQUET_PATH = os.path.join(BASE_DIR, "scripts", "interface", "tccs_dashboard_versao_tcc.parquet")

# ---- Funções utilitárias ----

def normalizar_texto(texto: str) -> str:
    """Remove acentuação, coloca em minúsculas e limpa espaços extras."""
    if not isinstance(texto, str):
        return ""
    texto = texto.strip().lower()
    texto = unicodedata.normalize("NFD", texto)
    texto = "".join(ch for ch in texto if unicodedata.category(ch) != "Mn")  # remove acentos
    return texto

def ler_cursos_parquet(path):
    """Lê o parquet e retorna lista de cursos únicos normalizados e originais."""
    df = pd.read_parquet(path)
    cursos = (
        df["curso"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    return cursos

def agrupar_cursos_localmente(cursos, limite_similaridade=85):
    """
    Agrupa cursos semelhantes localmente usando comparação fuzzy.
    Retorna um dicionário {curso_padrao: [variantes]}.
    """
    grupos = {}
    usados = set()

    for curso in cursos:
        if curso in usados:
            continue

        curso_norm = normalizar_texto(curso)
        grupo = [curso]
        usados.add(curso)

        for outro in cursos:
            if outro in usados:
                continue
            outro_norm = normalizar_texto(outro)
            similaridade = fuzz.token_sort_ratio(curso_norm, outro_norm)

            if similaridade >= limite_similaridade:
                grupo.append(outro)
                usados.add(outro)

        chave_grupo = normalizar_texto(grupo[0]).title()
        grupos[chave_grupo] = grupo

    return grupos

# ---- Execução principal ----
if __name__ == "__main__":
    print("📂 Lendo arquivo Parquet...")
    cursos = ler_cursos_parquet(PARQUET_PATH)
    print(f"📚 {len(cursos)} cursos encontrados.")

    print("🔍 Agrupando cursos semelhantes (processamento local)...")
    agrupamentos = agrupar_cursos_localmente(cursos)

    print("✅ Agrupamentos gerados:")
    print(f"📚 {len(agrupamentos)} cursos encontrados.")
    # print(json.dumps(agrupamentos, ensure_ascii=False, indent=2))

    # (opcional) salvar o resultado em arquivo JSON
    with open("agrupamentos_cursos.json", "w", encoding="utf-8") as f:
        json.dump(agrupamentos, f, ensure_ascii=False, indent=2)
        print("💾 Resultado salvo em agrupamentos_cursos.json")
