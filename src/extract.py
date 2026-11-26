"""
Módulo de Extracción (Capa Bronze) para el pipeline ETL de Pokémon TCG.

Implementa un flujo de extracción híbrido, desacoplado y resiliente:
1. Valida los archivos CSV existentes en `data/raw/` (incluyendo `pokemon-tcg-data-master 1999-2023.csv`
   y `tcg_prices_cache.csv`, o cualquier CSV con el esquema canónico).
2. Provee una interfaz desacoplada (`ExternalPriceProvider` / `PokemonTCGAPIProvider`) para
   fuentes externas futuras que lee credenciales opcionales desde `.env` sin fallar si no hay claves.
3. Si existe `pokemon-tcg-data-master 1999-2023.csv`, lo cruza con las cotizaciones reales en caché
   (o descargadas desde la API pública de Pokémon TCG) y mapea al esquema canónico.
4. Mantiene el generador sintético estocástico como fallback automático para pruebas o CI sin CSVs.
"""

import os
import glob
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Configuración del registrador de logs
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [EXTRACT]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Columnas canónicas requeridas para la capa Silver
REQUIRED_COLUMNS: List[str] = [
    "card_id",
    "name",
    "supertype",
    "subtype",
    "rarity",
    "artist",
    "set_id",
    "set_name",
    "series",
    "release_date",
    "market_price",
    "low_price",
    "high_price",
]

API_BASE_URL = "https://api.pokemontcg.io/v2/cards"
DEFAULT_CACHE_NAME = "tcg_prices_cache.csv"
DEFAULT_MASTER_NAME = "pokemon-tcg-data-master 1999-2023.csv"


# -----------------------------------------------------------------------------
# Interfaz desacoplada para futuras fuentes externas de precios
# -----------------------------------------------------------------------------
class ExternalPriceProvider(ABC):
    """Interfaz base desacoplada para proveedores externos de cotizaciones TCG."""

    @abstractmethod
    def is_configured(self) -> bool:
        """Indica si el proveedor cuenta con configuración/credenciales opcionales."""

    @abstractmethod
    def fetch_prices(self, cache_path: str) -> pd.DataFrame:
        """Obtiene cotizaciones de mercado y las retorna en formato tabular."""


class PokemonTCGAPIProvider(ExternalPriceProvider):
    """
    Proveedor para la API pública de Pokémon TCG (https://api.pokemontcg.io/v2/cards).
    Lee `POKEMON_TCG_API_KEY` desde variables de entorno/.env si existe, pero funciona
    sin clave (acceso público) y nunca falla por ausencia de credenciales.
    """

    def __init__(self, api_key: Optional[str] = None, page_size: int = 250, sleep_time: float = 0.15):
        self.api_key = api_key or os.getenv("POKEMON_TCG_API_KEY", "").strip()
        self.page_size = page_size
        self.sleep_time = sleep_time
        self.headers: Dict[str, str] = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        if self.api_key:
            self.headers["X-Api-Key"] = self.api_key

    def is_configured(self) -> bool:
        """Retorna True si hay API key en .env (el endpoint público funciona incluso en False)."""
        return bool(self.api_key)

    @staticmethod
    def extract_price_from_card(card: dict) -> Optional[Dict[str, Any]]:
        """
        Extrae market_price, low_price y high_price de un objeto de carta de la API.
        Prioriza tcgplayer.prices (holofoil o normal) y usa cardmarket como respaldo.
        """
        card_id = card.get("id")
        if not card_id:
            return None

        mp, lp, hp = None, None, None

        # 1. Prioridad: tcgplayer
        tcg = card.get("tcgplayer") or {}
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

        # 2. Respaldo: cardmarket
        cm = card.get("cardmarket") or {}
        cm_p = cm.get("prices") or {}
        if (mp is None or mp <= 0) and cm_p:
            mp = cm_p.get("averageSellPrice") or cm_p.get("trendPrice") or cm_p.get("avg30")
        if (lp is None or lp <= 0) and cm_p:
            lp = cm_p.get("lowPrice")
        if (hp is None or hp <= 0) and cm_p:
            hp = cm_p.get("trendPrice")

        # 3. Validación y consistencia lógica
        if mp is not None and mp > 0:
            mp_val = float(mp)
            lp_val = float(lp) if lp and lp > 0 else round(mp_val * 0.85, 2)
            hp_val = float(hp) if hp and hp > 0 else round(mp_val * 1.30, 2)
            lp_final = min(lp_val, mp_val)
            hp_final = max(hp_val, mp_val)

            return {
                "id": card_id,
                "market_price": round(mp_val, 2),
                "low_price": round(lp_final, 2),
                "high_price": round(hp_final, 2),
            }

        return None

    def _fetch_single_page(self, page: int, max_retries: int = 4) -> List[Dict[str, Any]]:
        """Descarga una página individual de la API con reintentos y tolerancia a fallos."""
        url = f"{API_BASE_URL}?page={page}&pageSize={self.page_size}&select=id,tcgplayer,cardmarket"
        for attempt in range(max_retries):
            try:
                r = requests.get(url, headers=self.headers, timeout=25)
                if r.status_code == 200:
                    data = r.json().get("data", [])
                    extracted = []
                    for card in data:
                        item = self.extract_price_from_card(card)
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

    def fetch_prices(self, cache_path: str) -> pd.DataFrame:
        """
        Gestiona la caché local de precios reales. Si no existe, descarga concurrentemente
        desde la API paginando y aplicando rate limits.
        """
        if os.path.exists(cache_path) and os.path.getsize(cache_path) > 100:
            logger.info(f"Caché de precios reales encontrada en '{cache_path}'. Cargando directamente...")
            df_cached = pd.read_csv(cache_path)
            logger.info(f"Cargadas {len(df_cached):,} cotizaciones reales desde la caché local.")
            return df_cached

        if not self.is_configured():
            logger.info("Sin POKEMON_TCG_API_KEY en .env; usando acceso público sin clave.")

        logger.info(f"No se encontró caché en '{cache_path}'. Iniciando descarga desde Pokémon TCG API...")
        os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)

        total_count = 20811
        try:
            r_init = requests.get(
                f"{API_BASE_URL}?page=1&pageSize={self.page_size}&select=id,tcgplayer,cardmarket",
                headers=self.headers,
                timeout=25,
            )
            if r_init.status_code == 200:
                total_count = r_init.json().get("totalCount", total_count)
        except Exception as e:
            logger.warning(f"No se pudo consultar totalCount en API ({e}); usando estimación por defecto.")

        total_pages = (total_count + self.page_size - 1) // self.page_size
        logger.info(f"Paginando {total_pages} páginas ({self.page_size} cartas/pág)...")

        all_prices: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_page = {
                executor.submit(self._fetch_single_page, p): p
                for p in range(1, total_pages + 1)
            }
            completed = 0
            for future in as_completed(future_to_page):
                p = future_to_page[future]
                try:
                    page_data = future.result()
                    all_prices.extend(page_data)
                    completed += 1
                    if completed % 15 == 0 or completed == total_pages:
                        logger.info(f"Progreso API: [{completed}/{total_pages} páginas] - {len(all_prices):,} precios capturados.")
                except Exception as e:
                    logger.warning(f"Error procesando página {p}: {e}")
                time.sleep(self.sleep_time)

        if not all_prices:
            logger.warning("No se pudieron descargar precios desde la API externa; retornando DataFrame vacío.")
            return pd.DataFrame(columns=["id", "market_price", "low_price", "high_price"])

        df_prices = pd.DataFrame(all_prices).drop_duplicates(subset=["id"], keep="last")
        df_prices.to_csv(cache_path, index=False)
        logger.info(f"Descarga finalizada. {len(df_prices):,} precios guardados en '{cache_path}'.")
        return df_prices


def validate_raw_csvs(raw_dir: str = "data/raw") -> Dict[str, Any]:
    """
    Valida los archivos CSV disponibles en `data/raw/`, verificando existencia,
    legibilidad y columnas presentes sin fallar si faltan archivos o claves API.
    """
    os.makedirs(raw_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    report: Dict[str, Any] = {
        "raw_dir": raw_dir,
        "csv_count": len(csv_files),
        "files": {},
        "has_master": False,
        "has_price_cache": False,
    }

    for path in csv_files:
        name = os.path.basename(path)
        try:
            sample = pd.read_csv(path, nrows=5, low_memory=False)
            report["files"][name] = {
                "valid": len(sample.columns) > 0,
                "columns": list(sample.columns),
                "size_bytes": os.path.getsize(path),
            }
            if name == DEFAULT_MASTER_NAME:
                report["has_master"] = True
            elif name == DEFAULT_CACHE_NAME:
                report["has_price_cache"] = True
        except Exception as exc:
            logger.warning(f"CSV inválido o ilegible '{name}': {exc}")
            report["files"][name] = {"valid": False, "error": str(exc)}

    logger.info(
        f"Validación Bronze en '{raw_dir}': {report['csv_count']} CSV(s) encontrados "
        f"(master={report['has_master']}, cache={report['has_price_cache']})."
    )
    return report


def load_master_dataset_with_prices(
    master_csv_path: str,
    cache_path: str,
    price_provider: Optional[ExternalPriceProvider] = None,
) -> pd.DataFrame:
    """
    Carga el CSV maestro de cartas y lo cruza con cotizaciones reales del proveedor externo o caché.
    """
    logger.info(f"Cargando dataset maestro real desde '{master_csv_path}'...")
    df_master = pd.read_csv(master_csv_path, low_memory=False)
    logger.info(f"Dataset maestro cargado: {len(df_master):,} cartas leídas.")

    provider = price_provider or PokemonTCGAPIProvider()
    df_prices = provider.fetch_prices(cache_path=cache_path)

    if df_prices.empty:
        logger.warning("Sin precios externos disponibles; activando respaldo sintético para mantener el pipeline operativo.")
        return generate_synthetic_pokemon_data(n_cards=2200)

    df_master["id_clean"] = df_master["id"].astype(str).str.strip()
    df_prices["id_clean"] = df_prices["id"].astype(str).str.strip()

    merged = pd.merge(df_master, df_prices, on="id_clean", how="inner")
    logger.info(f"Cruce completado: {len(merged):,} cartas con cotización real identificada.")

    merged = merged[merged["market_price"] > 0].copy()

    merged["card_id"] = merged["id_clean"]
    merged["set_id"] = merged["id_clean"].str.split("-").str[0]
    merged["set_name"] = merged["set"].fillna("Desconocido").astype(str).str.strip()
    merged["series"] = merged["series"].fillna("Desconocido").astype(str).str.strip()
    merged["release_date"] = merged["release_date"].fillna("2020-01-01").astype(str).str.strip()
    merged["rarity"] = merged["rarity"].fillna("Unknown").astype(str).str.strip()
    merged["artist"] = merged["artist"].fillna("Desconocido").astype(str).str.strip()
    merged["name"] = merged["name"].fillna("Desconocido").astype(str).str.strip()
    merged["supertype"] = merged["supertype"].fillna("Pokémon").astype(str).str.strip()

    merged["subtype"] = (
        merged["subtypes"]
        .astype(str)
        .str.replace("[", "", regex=False)
        .str.replace("]", "", regex=False)
        .str.replace("'", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )
    merged["subtype"] = merged["subtype"].replace({"": "Standard", "nan": "Standard"})

    final_df = merged[REQUIRED_COLUMNS].drop_duplicates(subset=["card_id"], keep="last")
    logger.info(f"Extracción y cruce finalizado: {len(final_df):,} cartas listas para Capa Silver.")
    return final_df


# -----------------------------------------------------------------------------
# Generador Sintético de Respaldo (Fallback para tests aislados y CI)
# -----------------------------------------------------------------------------
SETS_METADATA = [
    {"set_id": "base1", "set_name": "Base Set", "series": "Base", "release_date": "1999-01-09", "vintage_factor": 4.5},
    {"set_id": "base2", "set_name": "Jungle", "series": "Base", "release_date": "1999-06-16", "vintage_factor": 2.8},
    {"set_id": "base3", "set_name": "Fossil", "series": "Base", "release_date": "1999-10-10", "vintage_factor": 2.6},
    {"set_id": "base5", "set_name": "Team Rocket", "series": "Base", "release_date": "2000-04-24", "vintage_factor": 3.2},
    {"set_id": "neo1", "set_name": "Neo Genesis", "series": "Neo", "release_date": "2000-12-16", "vintage_factor": 3.0},
    {"set_id": "gym1", "set_name": "Gym Heroes", "series": "Gym", "release_date": "2000-08-14", "vintage_factor": 2.7},
    {"set_id": "neo2", "set_name": "Neo Discovery", "series": "Neo", "release_date": "2001-06-01", "vintage_factor": 2.5},
    {"set_id": "ex1", "set_name": "EX Ruby & Sapphire", "series": "EX", "release_date": "2003-06-18", "vintage_factor": 2.2},
    {"set_id": "ex8", "set_name": "EX Deoxys", "series": "EX", "release_date": "2005-02-14", "vintage_factor": 2.4},
    {"set_id": "dp1", "set_name": "Diamond & Pearl", "series": "Diamond & Pearl", "release_date": "2007-05-01", "vintage_factor": 1.7},
    {"set_id": "hgss1", "set_name": "HeartGold & SoulSilver", "series": "HeartGold & SoulSilver", "release_date": "2010-02-10", "vintage_factor": 1.9},
    {"set_id": "bw1", "set_name": "Black & White", "series": "Black & White", "release_date": "2011-04-25", "vintage_factor": 1.3},
    {"set_id": "xy2", "set_name": "Flashfire", "series": "XY", "release_date": "2014-05-07", "vintage_factor": 1.5},
    {"set_id": "sm1", "set_name": "Sun & Moon", "series": "Sun & Moon", "release_date": "2017-02-03", "vintage_factor": 1.1},
    {"set_id": "sm9", "set_name": "Team Up", "series": "Sun & Moon", "release_date": "2019-02-01", "vintage_factor": 1.8},
    {"set_id": "swsh1", "set_name": "Sword & Shield", "series": "Sword & Shield", "release_date": "2020-02-07", "vintage_factor": 1.0},
    {"set_id": "swsh7", "set_name": "Evolving Skies", "series": "Sword & Shield", "release_date": "2021-08-27", "vintage_factor": 2.1},
    {"set_id": "swsh9", "set_name": "Brilliant Stars", "series": "Sword & Shield", "release_date": "2022-02-25", "vintage_factor": 1.2},
    {"set_id": "sv1", "set_name": "Scarlet & Violet", "series": "Scarlet & Violet", "release_date": "2023-03-31", "vintage_factor": 1.0},
    {"set_id": "sv3pt5", "set_name": "151", "series": "Scarlet & Violet", "release_date": "2023-09-22", "vintage_factor": 1.6},
    {"set_id": "sv2", "set_name": "Paldea Evolved", "series": "Scarlet & Violet", "release_date": "2023-06-09", "vintage_factor": 1.1},
]

STANDARD_POKEMON_LIST = [
    "Bulbasaur", "Ivysaur", "Charmander", "Charmeleon", "Squirtle", "Wartortle",
    "Pikachu", "Raichu", "Gengar", "Alakazam", "Machamp", "Gyarados", "Eevee",
    "Snorlax", "Dragonite", "Mewtwo", "Mew", "Chikorita", "Cyndaquil", "Totodile",
]

ARTISTS = [
    "Ken Sugimori", "Mitsuhiro Arita", "Kouki Saitou", "5ban Graphics",
    "Naoki Saito", "sowsow", "AKIRA EGAWA", "PLANETA Otumami",
]


def generate_synthetic_pokemon_data(n_cards: int = 2200, seed: int = 42) -> pd.DataFrame:
    """Genera datos sintéticos con distribuciones realistas cuando no existen archivos crudos."""
    logger.info(f"Generando dataset sintético de respaldo ({n_cards} cartas)...")
    np.random.seed(seed)
    records = []
    cards_per_set = n_cards // len(SETS_METADATA)

    rarities = ["Common", "Uncommon", "Rare", "Rare Holo", "Rare Ultra", "Rare Secret"]
    rarity_weights = [0.45, 0.30, 0.14, 0.07, 0.03, 0.01]

    for s_meta in SETS_METADATA:
        for num in range(1, cards_per_set + 1):
            card_id = f"{s_meta['set_id']}-{num:03d}"
            name = np.random.choice(STANDARD_POKEMON_LIST)
            rarity = np.random.choice(rarities, p=rarity_weights)
            base_p = {
                "Common": 0.3, "Uncommon": 0.8, "Rare": 3.0,
                "Rare Holo": 12.0, "Rare Ultra": 35.0, "Rare Secret": 80.0
            }[rarity]
            market = round(max(0.10, base_p * np.random.lognormal(0, 0.5) * s_meta["vintage_factor"]), 2)
            low = round(market * np.random.uniform(0.75, 0.92), 2)
            high = round(market * np.random.uniform(1.10, 1.40), 2)

            records.append({
                "card_id": card_id, "name": name, "supertype": "Pokémon", "subtype": "Basic",
                "rarity": rarity, "artist": np.random.choice(ARTISTS), "set_id": s_meta["set_id"],
                "set_name": s_meta["set_name"], "series": s_meta["series"], "release_date": s_meta["release_date"],
                "market_price": market, "low_price": low, "high_price": high,
            })

    return pd.DataFrame(records)


def validate_raw_schema(df: pd.DataFrame, source_name: str = "Dataset") -> bool:
    """Valida presencia de columnas canónicas mínimas."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        logger.warning(f"Columnas faltantes en '{source_name}': {missing}")
        return False
    return True


def extract_raw_data(
    raw_dir: str = "data/raw",
    price_provider: Optional[ExternalPriceProvider] = None,
) -> pd.DataFrame:
    """
    Función principal de extracción (Capa Bronze):
    1. Valida los CSVs en `raw_dir`.
    2. Si existe `pokemon-tcg-data-master 1999-2023.csv`, lo cruza con cotizaciones reales (caché o API).
    3. Si no existe el CSV maestro, busca otros CSVs válidos o recurre al generador sintético.
    """
    validate_raw_csvs(raw_dir=raw_dir)
    master_csv_path = os.path.join(raw_dir, DEFAULT_MASTER_NAME)
    cache_path = os.path.join(raw_dir, DEFAULT_CACHE_NAME)

    if os.path.exists(master_csv_path):
        logger.info(f"Detectado dataset maestro real: '{master_csv_path}'. Iniciando extracción híbrida...")
        return load_master_dataset_with_prices(
            master_csv_path=master_csv_path,
            cache_path=cache_path,
            price_provider=price_provider,
        )

    csv_files = [f for f in glob.glob(os.path.join(raw_dir, "*.csv")) if DEFAULT_CACHE_NAME not in f]
    if csv_files:
        logger.info(f"Detectados {len(csv_files)} archivo(s) CSV alternativo(s).")
        dfs = []
        for f in csv_files:
            try:
                tdf = pd.read_csv(f, low_memory=False)
                if validate_raw_schema(tdf, source_name=os.path.basename(f)):
                    dfs.append(tdf)
            except Exception as e:
                logger.error(f"Error leyendo '{f}': {e}")
        if dfs:
            return pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["card_id"], keep="last")

    logger.info("No se encontraron CSVs maestros ni datos crudos válidos. Activando generador sintético...")
    return generate_synthetic_pokemon_data(n_cards=2200)


if __name__ == "__main__":
    df_result = extract_raw_data()
    print("\n--- Vista previa de cartas extraídas (Bronze) ---")
    print(df_result[["card_id", "name", "set_name", "rarity", "market_price", "low_price", "high_price"]].head(5))
    print(f"\nDimensiones finales: {df_result.shape}")
