import streamlit as st
import pandas as pd
import numpy as np
import pathlib as path
import math
import plotly.graph_objects as go
import json
import joblib
from battle import load_battle_data, build_features

st.set_page_config(page_title="Poké Coach", layout="wide")

RED, YELLOW, BLUE = "#EE1515", "#FFDE00", "#3B4CCA"

st.markdown(f"""
<style>
h1 {{ color: {YELLOW}; letter-spacing: 1px; }}
h2, h3 {{ border-left: 6px solid {RED}; padding-left: 10px; }}
[data-testid="stSidebar"] {{ border-right: 2px solid {BLUE}; }}
</style>
""", unsafe_allow_html=True)

DATA_DIR = path.Path(__file__).parent / "data"

MODEL_PATH = path.Path(__file__).parent / "models" / "win_model.joblib"

METRICS_PATH = path.Path(__file__).parent / "models" / "metrics.json"

POKEMON_CSV = DATA_DIR / "pokemon_complete.csv"

TYPES_CSV = DATA_DIR / "pokemon_types.csv"

STAT_COLS = ['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']

STAT_LABELS = {
    "hp": "HP", "attack": "Attack", "defense": "Defense",
    "sp_attack": "Sp. Atk", "sp_defense": "Sp. Def", "speed": "Speed",
}

def hex_to_rgba(hex_color, alpha):
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"

def stat_radar(a, b):
    labels = [STAT_LABELS[s] for s in STAT_COLS]
    top = max(max(a[s] for s in STAT_COLS), max(b[s] for s in STAT_COLS))
    rmax = max(150, math.ceil(top / 50) * 50)

    fig = go.Figure()
    for row, color in ((a, YELLOW), (b, RED)):
        vals = [int(row[s]) for s in STAT_COLS]
        fig.add_trace(go.Scatterpolar(
            r=vals + vals[:1],
            theta=labels + labels[:1],
            fill="toself",
            name=row["name"],
            line=dict(color=color),
            fillcolor=hex_to_rgba(color, 0.25),
        ))
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(range=[0, rmax], visible=True)),
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", y=-0.1),
    )
    return fig

ULTRA_BEASTS = {
    "Nihilego", "Buzzwole", "Pheromosa", "Xurkitree", "Celesteela",
    "Kartana", "Guzzlord", "Poipole", "Naganadel", "Stakataka", "Blacephalon",
}
PARADOX = {
    "Great-Tusk", "Scream-Tail", "Brute-Bonnet", "Flutter-Mane",
    "Slither-Wing", "Sandy-Shocks", "Roaring-Moon", "Walking-Wake",
    "Gouging-Fire", "Raging-Bolt", "Iron-Treads", "Iron-Bundle",
    "Iron-Hands", "Iron-Jugulis", "Iron-Moth", "Iron-Thorns",
    "Iron-Valiant", "Iron-Leaves", "Iron-Boulder", "Iron-Crown",
}

SPECIAL_NAMES = ULTRA_BEASTS | PARADOX

TYPE_COLORS = {
    "normal": "#A8A77A", "fire": "#EE8130", "water": "#6390F0",
    "electric": "#F7D02C", "grass": "#7AC74C", "ice": "#96D9D6",
    "fighting": "#C22E28", "poison": "#A33EA1", "ground": "#E2BF65",
    "flying": "#A98FF3", "psychic": "#F95587", "bug": "#A6B91A",
    "rock": "#B6A136", "ghost": "#735797", "dragon": "#6F35FC",
    "dark": "#705746", "steel": "#B7B7CE", "fairy": "#D685AD",
}

@st.cache_data
def load_pokemon() -> pd.DataFrame:
    df = pd.read_csv(POKEMON_CSV)
    df['type_1'] = df['type_1'].str.lower()
    df['type_2'] = df['type_2'].str.lower()
    df['type_2'] = df['type_2'].astype(object).where(df['type_2'].notna(), None)
    return df

@st.cache_data
def load_type_chart() -> pd.DataFrame:
    chart = pd.read_csv(TYPES_CSV)
    split_cols = [
        "double_damage_to", "half_damage_to", "no_damage_to",
        "double_damage_from", "half_damage_from", "no_damage_from",
    ]
    for col in split_cols:
        chart[col] = chart[col].fillna("").apply(
            lambda s: s.split("|") if s else []
        )
    return chart.set_index("type")

#--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

pokemon = load_pokemon()

type_chart = load_type_chart()

@st.cache_data
def build_effectiveness() -> dict:
    eff = {}
    for atk in type_chart.index:
        row = type_chart.loc[atk]
        for dfn in type_chart.index:
            if dfn in row["no_damage_to"]:
                eff[(atk, dfn)] = 0.0
            elif dfn in row["double_damage_to"]:
                eff[(atk, dfn)] = 2.0
            elif dfn in row["half_damage_to"]:
                eff[(atk, dfn)] = 0.5
            else:
                eff[(atk, dfn)] = 1.0
    return eff

EFF = build_effectiveness()

pokemon['is_form'] = pokemon['pokedex_number'] > 10000

st.sidebar.header('Recommendation Filters')
include_forms = st.sidebar.checkbox('Include Alternate Forms', value=False)
include_special = st.sidebar.checkbox('Include Legendary/Mythical Pokémon', value=False)
include_groups = st.sidebar.checkbox("Include Ultra Beasts & Paradox", value=False)

candidates = pokemon.copy()
if not include_forms:
    candidates = candidates[~candidates['is_form']]
if not include_special:
    candidates = candidates[~(candidates['is_legendary'] | candidates['is_mythical'])]
if not include_groups:
    candidates = candidates[~candidates['name'].isin(SPECIAL_NAMES)]

st.sidebar.caption(f"({len(candidates)} Pokémon available)")

#--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

def type_multiplier(atk_type, def_types):
    mult = 1.0
    for t in def_types:
        if t is not None and pd.notna(t):
            mult *= EFF[(atk_type, t)]
    return mult

def type_badge(t):
    return (
        f"<span style='background: {TYPE_COLORS[t]}; color: white;"
        f"padding:2px 10px;border-radius:10px;font-size:0.8rem;"
        f"margin-right:4px'>{t.title()}</span>"
    )

DEFAULTS = {
    "w_offense": 3.0,
    "w_defense": 3.0,
    "w_speed": 1.0,
    "w_bulk": 1.0,
    "w_power": 1.0,
    "weakest_link": 0.4,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

def reset_weights():
    for k, v in DEFAULTS.items():
        st.session_state[k] = v

st.sidebar.header("Ranking weights")
WEIGHTS = {
    "offense": st.sidebar.slider(
        "Offense (type advantage)", min_value=0.0, max_value=5.0, step=0.5,
        key="w_offense",
        help="How much hitting the opponent super effectively matters."),
    "defense": st.sidebar.slider(
        "Defense (resisting their attacks)", min_value=0.0, max_value=5.0,
        step=0.5, key="w_defense",
        help="How much resisting the opponent's types matters."),
    "speed": st.sidebar.slider(
        "Speed (moving first)", min_value=0.0, max_value=5.0, step=0.5,
        key="w_speed",
        help="Bonus for having a higher Speed stat than the opponent."),
    "bulk": st.sidebar.slider(
        "Bulk (HP + defenses)", min_value=0.0, max_value=5.0, step=0.5,
        key="w_bulk",
        help="Bonus for being able to take hits."),
    "power": st.sidebar.slider(
        "Power (attack stats)", min_value=0.0, max_value=5.0, step=0.5,
        key="w_power",
        help="Bonus for a high Attack or Sp. Attack stat."),
}
WEAKEST_LINK = st.sidebar.slider(
    "Team mode: weakest-link weight", min_value=0.0, max_value=1.0, step=0.1,
    key="weakest_link",
    help="0 = only the average across opponents counts. "
         "1 = only the worst matchup counts.")

st.sidebar.button("Reset to defaults", on_click=reset_weights)

st.sidebar.header("Ranking method")
methods = ["Formula", "ML model"] if MODEL_PATH.exists() else ["Formula"]
method = st.sidebar.radio("Rank with", methods)
if method == "ML model":
    st.sidebar.caption("Weight sliders only affect the Formula method. "
                       "The weakest-link slider still applies.")

def types_of(row):
    return [t for t in (row["type_1"], row["type_2"]) if t is not None and pd.notna(t)]

def log_mult(m):
    # 0.25x -> -2, 0.5x -> -1, 1x -> 0, 2x -> 1, 4x -> 2 (immune counts as -2)
    return np.log2(max(m, 0.25))

def score_matchup(c, o):
    c_types, o_types = types_of(c), types_of(o)
    offense = log_mult(max(type_multiplier(t, o_types) for t in c_types))
    danger = log_mult(max(type_multiplier(t, c_types) for t in o_types))
    speed = 1.0 if c["speed"] > o["speed"] else 0.0
    bulk = (c["hp"] * c["defense"] * c["sp_defense"]) ** (1 / 3) / 100
    power = max(c["attack"], c["sp_attack"]) / 100
    return (WEIGHTS["offense"] * offense
            - WEIGHTS["defense"] * danger
            + WEIGHTS["speed"] * speed
            + WEIGHTS["bulk"] * bulk
            + WEIGHTS["power"] * power)

def fmt_mult(m):
    return {0.0: "0", 0.25: "¼", 0.5: "½", 1.0: "1", 2.0: "2", 4.0: "4"}.get(m, str(m))

def explain_single(c, o):
    c_types, o_types = types_of(c), types_of(o)

    best_atk = max(c_types, key=lambda t: type_multiplier(t, o_types))
    atk_mult = type_multiplier(best_atk, o_types)

    worst_def = max(o_types, key=lambda t: type_multiplier(t, c_types))
    def_mult = type_multiplier(worst_def, c_types)

    if atk_mult > 0:
        atk_txt = f"Hits {fmt_mult(atk_mult)}× with {best_atk.title()}"
    else:
        atk_txt = "Can't damage it with its own types"

    if def_mult == 0:
        def_txt = f"Immune to {worst_def.title()}"
    else:
        def_txt = f"Takes {fmt_mult(def_mult)}× from {worst_def.title()}"

    speed_txt = "Outspeeds" if c["speed"] > o["speed"] else "Slower"
    return f"{atk_txt}. {def_txt}. {speed_txt}."

def explain_team(c, opp_rows, scores):
    best = opp_rows[int(np.argmax(scores))]["name"]
    worst = opp_rows[int(np.argmin(scores))]["name"]
    super_eff = sum(
        1 for o in opp_rows
        if max(type_multiplier(t, types_of(o)) for t in types_of(c)) >= 2
    )
    return (f"Super effective vs {super_eff} of {len(opp_rows)}. "
            f"Best vs {best}, weakest vs {worst}.")

def rank_candidates(opps, pool, n=15):
    opp_rows = [r for _, r in opps.iterrows()]
    results = []
    for _, c in pool.iterrows():
        scores = [score_matchup(c, o) for o in opp_rows]
        final = (1 - WEAKEST_LINK) * np.mean(scores) + WEAKEST_LINK * min(scores)
        results.append({
            "final": final,
            "Sprite": c["sprite_url"] if pd.notna(c["sprite_url"]) else None,
            "Name": c["name"],
            "Types": " / ".join(t.title() for t in types_of(c)),
            "Score": round(final, 1),
            "_row": c,
            "_scores": scores,
        })

    results.sort(key=lambda r: r["final"], reverse=True)
    top = results[:n]

    rows = []
    for i, r in enumerate(top, start=1):
        if len(opp_rows) == 1:
            why = explain_single(r["_row"], opp_rows[0])
        else:
            why = explain_team(r["_row"], opp_rows, r["_scores"])
        rows.append({"Rank": i, "Sprite": r["Sprite"], "Name": r["Name"],
                     "Types": r["Types"], "Score": r["Score"], "Why": why})
    return pd.DataFrame(rows)


@st.cache_resource
def load_ml():
    model = joblib.load(MODEL_PATH)
    bdf, barr = load_battle_data(include_forms=True)
    name_to_idx = {n: i for i, n in enumerate(bdf["name"])}
    return model, barr, name_to_idx


def rank_candidates_ml(opps, pool, n=15):
    model, barr, name_to_idx = load_ml()
    opp_rows = [r for _, r in opps.iterrows()]
    opp_idx = np.array([name_to_idx[x] for x in opps["name"]])
    cand_idx = np.array([name_to_idx[x] for x in pool["name"]])

    # every candidate vs every opponent, predicted in one batch
    ci = np.repeat(cand_idx, len(opp_idx))
    oi = np.tile(opp_idx, len(cand_idx))
    probs = model.predict_proba(build_features(barr, ci, oi))[:, 1]
    probs = probs.reshape(len(cand_idx), len(opp_idx))

    final = (1 - WEAKEST_LINK) * probs.mean(axis=1) + WEAKEST_LINK * probs.min(axis=1)
    order = np.argsort(-final)[:n]

    rows = []
    for rank, k in enumerate(order, start=1):
        c = pool.iloc[k]
        p = probs[k]
        if len(opp_rows) == 1:
            why = f"{p[0]:.0%} predicted win chance. " + explain_single(c, opp_rows[0])
        else:
            best_o = opp_rows[int(np.argmax(p))]["name"]
            worst_o = opp_rows[int(np.argmin(p))]["name"]
            why = (f"Predicted to win {int((p > 0.5).sum())} of {len(p)}. "
                   f"Best vs {best_o}, weakest vs {worst_o} ({p.min():.0%}).")
        rows.append({
            "Rank": rank,
            "Sprite": c["sprite_url"] if pd.notna(c["sprite_url"]) else None,
            "Name": c["name"],
            "Types": " / ".join(t.title() for t in types_of(c)),
            "Score": round(final[k] * 100, 1),
            "Why": why,
        })
    return pd.DataFrame(rows)

st.title("Poké Coach")
st.write('Poke Coach is here to help you strategically pick the best Pokémon for your team.') 
st.caption('(NOT AFFILIATED WITH NINTENDO/POKÉMON - COPYRIGHTS BELONG TO THEIR RESPECTIVE OWNERS)')

#--- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

st.subheader("Who are you up against?")

names = pokemon.sort_values("pokedex_number")["name"].tolist()
selected = st.multiselect(
    "Pick up to 6 opposing Pokémon",
    names,
    max_selections=6,
    placeholder="Start typing a name...",
)

opponents = pokemon[pokemon["name"].isin(selected)]
opponents = opponents.set_index("name").loc[selected].reset_index()

if opponents.empty:
    st.info("Pick at least one opposing Pokémon to get recommendations.")
else:
    cols = st.columns(len(opponents))
    for col, (_, row) in zip(cols, opponents.iterrows()):
        with col:
            if pd.notna(row["sprite_url"]):
                st.image(row["sprite_url"], width=96)
            st.markdown(f"**{row['name']}**")
            types = [t for t in (row["type_1"], row["type_2"]) if pd.notna(t)]
            st.markdown("".join(type_badge(t) for t in types),
                        unsafe_allow_html=True)


st.subheader("Top 15 Recommendations")

if opponents.empty:
    st.caption("Waiting for opponents...")
else:
    pool = candidates[~candidates["name"].isin(opponents["name"])]
    if method == "ML model":
        top15 = rank_candidates_ml(opponents, pool)
    else:
        top15 = rank_candidates(opponents, pool)
    best = top15.iloc[0]
    c1, c2 = st.columns([1, 5])
    with c1:
        if pd.notna(best["Sprite"]):
            st.image(best["Sprite"], width=120)
    with c2:
        st.markdown(
            f"<div style='border-left:6px solid {YELLOW};padding-left:12px'>"
            f"<div style='color:{YELLOW};font-weight:700'>TOP PICK</div>"
            f"<div style='font-size:1.8rem;font-weight:700'>{best['Name']}</div>"
            f"<div>{best['Types']} · Score {best['Score']}</div>"
            f"<div>{best['Why']}</div></div>",
            unsafe_allow_html=True,
        )
    st.dataframe(
        top15,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Sprite": st.column_config.ImageColumn("Sprite"),
            "Why": st.column_config.TextColumn("Why", width="large"),
        },
    )
    st.subheader("Stat comparison")
    col_a, col_b = st.columns(2)
    pick_name = col_a.selectbox("Recommended Pokémon", top15["Name"].tolist())
    opp_name = col_b.selectbox("Compare against", opponents["name"].tolist())

    a = pokemon[pokemon["name"] == pick_name].iloc[0]
    b = pokemon[pokemon["name"] == opp_name].iloc[0]

    st.plotly_chart(stat_radar(a, b), use_container_width=True)
    st.caption(f"Base stat total: {pick_name} {a['base_stat_total']} "
               f"vs {opp_name} {b['base_stat_total']}")

with st.expander("How accurate is each method?"):
    if METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        m1, m2, m3 = st.columns(3)
        m1.metric("Hand-built formula", f"{m['formula_accuracy']:.1%}")
        m2.metric("Logistic regression", f"{m['logistic_accuracy']:.1%}")
        m3.metric("Gradient boosting (ML model)", f"{m['gradient_boost_accuracy']:.1%}")
        st.caption(
            f"Accuracy at picking the winner of simulated 1v1 matchups, tested on "
            f"{m['test_pairs']:,} matchups between Pokémon the model never saw in "
            f"training. The simulator is simplified: level 50, one 80-power "
            f"same-type move each, no abilities, items or held moves."
        )
    else:
        st.caption("Run train_model.py to generate accuracy numbers.")