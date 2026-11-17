"""The two knowledge sources the assistant can use.

* **Table**: "Fréquentation des musées de France 2018" (Ministère de la Culture, Licence
  Ouverte / Etalab): one row per "Musée de France" with paid and free admissions.
* **Documents**: PIAF (Etalab, MIT licence): 761 French Wikipedia paragraphs with 3,835
  questions written and answered by volunteers.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
MUSEUMS_CSV = "frequentation-musees-de-france-2018.csv"
PIAF = "piaf.parquet"

REGIONS = {
    "Auvergne-Rhône-Alpes": "01 03 07 15 26 38 42 43 63 69 73 74",
    "Bourgogne-Franche-Comté": "21 25 39 58 70 71 89 90",
    "Bretagne": "22 29 35 56",
    "Centre-Val de Loire": "18 28 36 37 41 45",
    "Corse": "2A 2B 20",
    "Grand Est": "08 10 51 52 54 55 57 67 68 88",
    "Hauts-de-France": "02 59 60 62 80",
    "Île-de-France": "75 77 78 91 92 93 94 95",
    "Normandie": "14 27 50 61 76",
    "Nouvelle-Aquitaine": "16 17 19 23 24 33 40 47 64 79 86 87",
    "Occitanie": "09 11 12 30 31 32 34 46 48 65 66 81 82",
    "Pays de la Loire": "44 49 53 72 85",
    "Provence-Alpes-Côte d'Azur": "04 05 06 13 83 84",
    "Guadeloupe": "971",
    "Martinique": "972",
    "Guyane": "973",
    "La Réunion": "974",
    "Mayotte": "976",
}
DEPT_TO_REGION = {d: r for r, depts in REGIONS.items() for d in depts.split()}


def _department(museum_id: str) -> str:
    museum_id = str(museum_id).strip()
    if museum_id[:2] == "97":
        return museum_id[:3]
    return museum_id[:2]


def load_museums(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    raw = pd.read_csv(data_dir / MUSEUMS_CSV, dtype=str, keep_default_na=False)
    stats = raw["stats"].str.extractall(r"(?P<k>[\w-]+):(?P<v>[^;]*)").reset_index()
    wide = stats.pivot(index="level_0", columns="k", values="v")
    df = pd.DataFrame(
        {
            "id": raw["id"],
            "nom": raw["name"].str.strip(),
            "ville": raw["city"].str.strip().str.title(),
            "departement": raw["id"].map(_department),
        }
    )
    df["region"] = df["departement"].map(DEPT_TO_REGION).fillna("Inconnue")
    df["entrees_payantes"] = pd.to_numeric(wide["payant"], errors="coerce").fillna(0).astype(int)
    df["entrees_gratuites"] = pd.to_numeric(wide["gratuit"], errors="coerce").fillna(0).astype(int)
    df["entrees_totales"] = df["entrees_payantes"] + df["entrees_gratuites"]
    df["annee"] = 2018
    return df


TABLE_SCHEMA = """CREATE TABLE musees (
  id TEXT,                 -- identifiant du musée
  nom TEXT,                -- nom du musée
  ville TEXT,              -- commune (ex. 'Lyon', 'Paris')
  departement TEXT,        -- code département (ex. '69', '2A', '974')
  region TEXT,             -- région (ex. 'Bretagne', 'Île-de-France', 'Occitanie')
  entrees_payantes INTEGER,
  entrees_gratuites INTEGER,
  entrees_totales INTEGER, -- payantes + gratuites
  annee INTEGER            -- toujours 2018
);"""


def connect(museums: pd.DataFrame) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    museums.to_sql("musees", conn, index=False)
    return conn


def load_piaf(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    df = pd.read_parquet(data_dir / PIAF)
    df["answer"] = df["answer"].map(list)
    return df


def piaf_corpus(piaf: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct paragraph, with a stable id."""
    corpus = piaf[["title", "context"]].drop_duplicates("context").reset_index(drop=True)
    corpus["doc_id"] = [f"p{i:04d}" for i in range(len(corpus))]
    return corpus
