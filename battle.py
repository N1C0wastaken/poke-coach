from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
STATS = ["hp", "attack", "defense", "sp_attack", "sp_defense", "speed"]


def load_battle_data(include_forms=False):
    df = pd.read_csv(DATA_DIR / "pokemon_complete.csv")
    if not include_forms:
        df = df[df["pokedex_number"] <= 10000]
    df = df.reset_index(drop=True)
    df["type_1"] = df["type_1"].str.lower()
    df["type_2"] = df["type_2"].str.lower()

    chart = pd.read_csv(DATA_DIR / "pokemon_types.csv")
    for col in chart.columns[1:]:
        chart[col] = chart[col].fillna("").apply(lambda s: s.split("|") if s else [])
    chart = chart.set_index("type")

    types = list(chart.index)
    idx = {t: i for i, t in enumerate(types)}
    mult = np.ones((len(types), len(types)))
    for a in types:
        row = chart.loc[a]
        for d in types:
            if d in row["no_damage_to"]:
                mult[idx[a], idx[d]] = 0.0
            elif d in row["double_damage_to"]:
                mult[idx[a], idx[d]] = 2.0
            elif d in row["half_damage_to"]:
                mult[idx[a], idx[d]] = 0.5

    arrays = {
        "t1": df["type_1"].map(idx).values,
        "t2": df["type_2"].map(idx).fillna(-1).astype(int).values,
        "stats": {c: df[c].values.astype(float) for c in STATS},
        "mult": mult,
    }
    return df, arrays


def _mult_vs(arr, att_types, defender):
    m = arr["mult"][att_types, arr["t1"][defender]].copy()
    has2 = arr["t2"][defender] >= 0
    m[has2] *= arr["mult"][att_types[has2], arr["t2"][defender][has2]]
    return m


def best_stab(arr, i, j):
    """Best STAB multiplier of Pokemon i (index array) against Pokemon j."""
    first = _mult_vs(arr, arr["t1"][i], j)
    has2 = arr["t2"][i] >= 0
    second = np.where(has2, _mult_vs(arr, np.where(has2, arr["t2"][i], 0), j), -1)
    return np.maximum(first, second)


def damage(arr, i, j, mult):
    s = arr["stats"]
    phys = s["attack"][i] >= s["sp_attack"][i]
    atk = np.where(phys, s["attack"][i], s["sp_attack"][i]) + 5
    dfn = np.where(phys, s["defense"][j], s["sp_defense"][j]) + 5
    base = np.floor(22 * 80 * atk / dfn / 50) + 2
    return np.floor(base * 1.5 * mult)


def simulate(arr, i, j, rng):
    """1v1 at level 50, one 80-power STAB move each. Returns (i_wins, is_draw)."""
    s = arr["stats"]
    d1 = damage(arr, i, j, best_stab(arr, i, j))
    d2 = damage(arr, j, i, best_stab(arr, j, i))
    hp1, hp2 = s["hp"][i] + 60, s["hp"][j] + 60
    k1 = np.where(d1 > 0, np.ceil(hp2 / np.maximum(d1, 1)), np.inf)
    k2 = np.where(d2 > 0, np.ceil(hp1 / np.maximum(d2, 1)), np.inf)
    tie = rng.random(len(i)) < 0.5
    i_first = (s["speed"][i] > s["speed"][j]) | ((s["speed"][i] == s["speed"][j]) & tie)
    win = np.where(k1 < k2, 1, np.where(k1 > k2, 0, np.where(i_first, 1, 0)))
    draw = np.isinf(k1) & np.isinf(k2)
    return win, draw


def build_features(arr, i, j):
    s = arr["stats"]
    f = {"atk_mult": best_stab(arr, i, j), "def_mult": best_stab(arr, j, i),
         "speed_diff": s["speed"][i] - s["speed"][j]}
    for c in STATS:
        f["a_" + c] = s[c][i]
        f["b_" + c] = s[c][j]
    return pd.DataFrame(f)