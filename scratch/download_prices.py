"""
Script utilitario para descargar y cachear precios reales desde la API de Pokémon TCG.
"""

import os
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
import pandas as pd

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("PRICES_FETCHER")

CACHE_PATH = "data/raw/tcg_prices_cache.csv"
BASE_URL = "https://api.pokemontcg.io/v2/cards"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}


def extract_price_from_card(c: dict) -> dict:
    """Extrae market_price, low_price y high_price de un objeto de carta."""
    card_id = c.get("id")
    if not card_id:
        return None

    mp, lp, hp = None, None, None

    # 1. Priorizar tcgplayer
    tcg = c.get("tcgplayer") or {}
    prices = tcg.get("prices") or {}
    
    preferred_variants = [
        "holofoil", "normal", "reverseHolofoil", "unlimitedHolofoil",
        "1stEditionHolofoil", "1stEditionNormal"
    ]
    chosen = None
    for var in preferred_variants:
        if var in prices and isinstance(prices[var], dict):
            chosen = prices[var]
            break

    if not chosen and prices:
        first = list(prices.values())[0]
        if isinstance(first, dict):
            chosen = first

    if chosen:
        mp = chosen.get("market") or chosen.get("mid")
        lp = chosen.get("low") or chosen.get("directLow")
        hp = chosen.get("high")

    # 2. Respaldo en cardmarket
    cm = c.get("cardmarket") or {}
    cm_p = cm.get("prices") or {}
    if (mp is None or mp <= 0) and cm_p:
        mp = cm_p.get("averageSellPrice") or cm_p.get("trendPrice") or cm_p.get("avg30")
    if (lp is None or lp <= 0) and cm_p:
        lp = cm_p.get("lowPrice")
    if (hp is None or hp <= 0) and cm_p:
        hp = cm_p.get("trendPrice")

    # 3. Consistencia
    if mp and mp > 0:
        mp_val = float(mp)
        lp_val = float(lp) if lp and lp > 0 else round(mp_val * 0.85, 2)
        hp_val = float(hp) if hp and hp > 0 else round(mp_val * 1.30, 2)
        lp_final = min(lp_val, mp_val)
        hp_final = max(hp_val, mp_val)

        return {
            "id": card_id,
            "market_price": round(mp_val, 2),
            "low_price": round(lp_final, 2),
            "high_price": round(hp_final, 2)
        }

    return None


def fetch_single_page(page: int, page_size: int = 250, max_retries: int = 4) -> list:
    """Descarga una página específica con reintentos exponenciales."""
    url = f"{BASE_URL}?page={page}&pageSize={page_size}&select=id,tcgplayer,cardmarket"
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=25)
            if r.status_code == 200:
                data = r.json().get("data", [])
                extracted = []
                for card in data:
                    item = extract_price_from_card(card)
                    if item:
                        extracted.append(item)
                return extracted
            elif r.status_code == 429:
                time.sleep(2.0 * (attempt + 1))
            else:
                time.sleep(1.0 * (attempt + 1))
        except Exception:
            time.sleep(1.0 * (attempt + 1))
    return []


def download_all_prices():
    """Descarga paginada de todos los precios reales con concurrencia moderada y barra de progreso."""
    os.makedirs("data/raw", exist_ok=True)
    if os.path.exists(CACHE_PATH):
        logger.info(f"Caché ya existente en {CACHE_PATH}. Omitiendo descarga.")
        return

    logger.info("Iniciando descarga de precios reales desde api.pokemontcg.io...")
    # Obtener el conteo total en la primera página
    first_res = requests.get(f"{BASE_URL}?page=1&pageSize=250&select=id,tcgplayer,cardmarket", headers=HEADERS, timeout=25)
    if first_res.status_code == 200:
        j = first_res.json()
        total_count = j.get("totalCount", 20811)
    else:
        total_count = 20811

    page_size = 250
    total_pages = (total_count + page_size - 1) // page_size
    logger.info(f"Total de cartas estimadas: {total_count:,} en {total_pages} páginas de {page_size}.")

    all_prices = []
    # Descargar usando ThreadPoolExecutor con 3 workers para respetar rate limits y maximizar velocidad
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_page = {executor.submit(fetch_single_page, p, page_size): p for p in range(1, total_pages + 1)}
        completed = 0
        for future in as_completed(future_to_page):
            p = future_to_page[future]
            try:
                page_data = future.result()
                all_prices.extend(page_data)
                completed += 1
                if completed % 10 == 0 or completed == total_pages:
                    logger.info(f"Progreso: [{completed}/{total_pages} páginas] - {len(all_prices):,} cartas con precio real.")
            except Exception as e:
                logger.warning(f"Error procesando página {p}: {e}")

    df_prices = pd.DataFrame(all_prices).drop_duplicates(subset=["id"], keep="last")
    df_prices.to_csv(CACHE_PATH, index=False)
    logger.info(f"Descarga finalizada. {len(df_prices):,} precios guardados en {CACHE_PATH}.")


if __name__ == "__main__":
    download_all_prices()
