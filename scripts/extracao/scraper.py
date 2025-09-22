# C:\...\extracao\scraper.py

import asyncio
import aiohttp
import time
from datetime import datetime

import config  # Importação direta
from database import DatabaseManager, clean_value # Importação direta

def log(msg):
    """Função simples de log com timestamp."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

# O restante do código deste arquivo permanece exatamente o mesmo da resposta anterior...
# (fetch_professores, _fetch_detail, fetch_detalhes, run_for_institution)
async def fetch_professores(sigla, base_url, db_manager, progress_callback=None):
    """Busca a lista de todos os professores de uma instituição."""
    list_url = f"{base_url}/api/portfolio/pessoa/data"
    professores = []
    start = 0
    total = 1

    if progress_callback:
        progress_callback(0, "?")

    async with aiohttp.ClientSession() as session:
        while True:
            params = {"start": start, "length": config.PAGE_SIZE}
            try:
                async with session.get(list_url, params=params, headers=config.DEFAULT_HEADERS, ssl=False) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
            except Exception as e:
                log(f"[{sigla}] Erro ao buscar lista de professores (start={start}): {e}")
                break

            if not isinstance(data, list) or len(data) < 2 or not data[1]:
                break

            meta, batch = data[0] or {}, data[1] or []
            total = meta.get("total", total)
            length_returned = meta.get("length", len(batch)) or len(batch)

            for p in batch:
                if slug := p.get("slug"):
                    professores.append({
                        "nome": p.get("nome"),
                        "campus": p.get("campusNome"),
                        "cargo": p.get("cargo"),
                        "slug": slug,
                        "url_final": f"{base_url}/portfolio/pessoas/{slug}",
                    })
            
            log(f"[{sigla}] [{len(professores)}/{total}] - professores coletados")
            if progress_callback:
                progress_callback(len(professores), total)

            if len(professores) >= total:
                break
            
            start += length_returned
            await asyncio.sleep(0.05)

    if professores:
        db_manager.save_professores(sigla, professores)
    return professores

async def _fetch_detail(session, detail_url, p):
    """Função auxiliar para buscar o detalhe de um único professor."""
    slug = p["slug"]
    start_time = time.perf_counter()
    try:
        async with session.get(f"{detail_url}/{slug}", headers=config.DEFAULT_HEADERS, ssl=False) as resp:
            resp.raise_for_status()
            data = await resp.json()
            elapsed = time.perf_counter() - start_time
            return slug, p, data, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return slug, p, {"erro": str(e)}, elapsed

async def fetch_detalhes(sigla, base_url, uf, professores, db_manager, progress_callback=None):
    """Busca os detalhes (TCCs) para uma lista de professores."""
    if not professores:
        log(f"[{sigla}] Nenhum professor para buscar detalhes.")
        return

    detail_url = f"{base_url}/api/portfolio/pessoa/s"
    connector = aiohttp.TCPConnector(limit=config.MAX_CONCURRENT)
    completed = 0
    total = len(professores)

    log(f"[{sigla}] Coletando detalhes de {total} professores...")
    if progress_callback:
        progress_callback(0, total)

    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [_fetch_detail(session, detail_url, {**p, "sigla": sigla}) for p in professores]
        
        for coro in asyncio.as_completed(tasks):
            slug, prof, data, elapsed = await coro
            completed += 1
            
            log(f"[{sigla}] [{completed}/{total}] - {slug} -> {elapsed:.2f}s")
            if progress_callback:
                progress_callback(completed, total)

            tccs_para_salvar = []
            outra_producao = data.get("outraProducao", {})
            if isinstance(outra_producao, dict) and "orientacoesConcluidas" in outra_producao:
                for item in outra_producao.get("orientacoesConcluidas", []):
                    for trabalho in item.get("outrasOrientacoesConcluidas", []):
                        dados_basicos = trabalho.get("dadosBasicosDeOutrasOrientacoesConcluidas", {})
                        if dados_basicos.get("natureza") != "TRABALHO_DE_CONCLUSAO_DE_CURSO_GRADUACAO":
                            continue

                        detalhamento = trabalho.get("detalhamentoDeOutrasOrientacoesConcluidas", {})
                        nome_professor = clean_value(prof.get("nome"))
                        autores = clean_value(detalhamento.get("nomeDoOrientado"))
                        if nome_professor:
                            autores = (autores + ", " if autores else "") + f"{nome_professor} (Orientador/a)"

                        palavras = trabalho.get("palavrasChave") or {}
                        info_add = trabalho.get("informacoesAdicionais") or {}

                        tccs_para_salvar.append((
                            slug, nome_professor, prof.get("sigla"),
                            clean_value(detalhamento.get("nomeDaInstituicao")), uf, clean_value(prof.get("campus")),
                            clean_value(dados_basicos.get("ano")), clean_value(detalhamento.get("nomeDoCurso")),
                            autores, clean_value(dados_basicos.get("titulo")),
                            clean_value(info_add.get("descricaoInformacoesAdicionais")),
                            clean_value(palavras.get("palavrasChaves"))
                        ))
            
            if tccs_para_salvar:
                db_manager.save_tccs(tccs_para_salvar)
    
    log(f"[{sigla}] Todos os TCCs salvos.")

async def run_for_institution(sigla, base_url, uf, db_manager, callbacks):
    """Executa o pipeline completo para uma instituição."""
    log(f"=== {sigla}: Iniciando coleta ===")
    
    professores = await fetch_professores(sigla, base_url, db_manager, callbacks.get('prof_progress'))
    log(f"[{sigla}] Total de professores encontrados: {len(professores)}")
    
    await fetch_detalhes(sigla, base_url, uf, professores, db_manager, callbacks.get('det_progress'))
    
    log(f"=== {sigla}: Coleta concluída ===")
