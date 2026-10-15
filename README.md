# Poké Coach

**Live app:** https://poke-coach.streamlit.app

Pick the Pokémon you're up against (one or a whole team of up to six) and Poké Coach ranks the top 15 Pokémon to use against them, with a plain-language reason for each pick.

> Fan-made, non-commercial portfolio project. Not affiliated with or endorsed by Nintendo, Game Freak, or The Pokémon Company. Pokémon and its names are trademarks of their respective owners.

## Features

- Opponent picker with sprites and type badges, up to 6 opponents at once
- Top 15 recommendations with a "Why" column (type advantage, resistances, speed)
- Two ranking methods you can switch between: a hand-built formula and a machine-learning model
- Adjustable formula weights (offense, defense, speed, bulk, power) with a reset button
- Team mode with a "weakest link" slider: average across all opponents vs. worst matchup
- Filters for alternate forms, legendary/mythical Pokémon, and Ultra Beasts / Paradox Pokémon
- Radar chart comparing base stats of a recommended Pokémon and an opponent
- Dark theme in Pokémon red, yellow and blue

## How it works

**Formula method.** For every candidate, the app scores the matchup against each opponent using:

- Offense: the candidate's best same-type attack multiplier against the opponent
- Defense: the opponent's best same-type attack multiplier against the candidate (lower is better)
- Speed, bulk (HP, Defense, Sp. Defense) and power (Attack or Sp. Attack)

Multipliers are put on a log scale (0.25x to 4x becomes -2 to +2) and combined with adjustable weights. In team mode, the final score blends the average and the worst matchup.

**ML method.** I wrote a simplified 1v1 battle simulator (level 50, one 80-power same-type move each, faster Pokémon moves first). It was run on about 300,000 random matchups to create labeled data. A gradient-boosted classifier (scikit-learn) learns to predict the winner from type multipliers and base stats, and the app ranks candidates by predicted win probability.

## Results

Accuracy at picking the winner of simulated matchups, tested on 11,972 matchups between Pokémon the model never saw during training (20% of species were held out; the training set had 191,415 matchups):

| Method | Accuracy |
|---|---|
| Hand-built formula | 78.9% |
| Logistic regression | 90.0% |
| Gradient boosting (used in app) | 94.0% |

The learned model beat the hand-tuned weights by about 15 percentage points.

## Limitations

- The "ground truth" comes from my simplified simulator, not real games. There are no movesets, abilities, items, status effects or switching, so these are recommendations based on types and base stats, not guarantees for competitive play.
- Pokémon that depend heavily on abilities or moves can be over- or under-rated.
- The deployed app retrains the model on startup if the saved model file can't be loaded (for example, after a scikit-learn version change). The accuracy numbers above come from the held-out evaluation in `train_model.py`.

## Tech stack

Python, pandas, NumPy, scikit-learn, Streamlit, Plotly, joblib

## Project structure

```
PokeCoach/
├── app.py              # Streamlit app (UI, formula ranking, ML ranking)
├── battle.py           # Battle simulator and feature builder
├── train_model.py      # Simulates matchups, trains and evaluates models
├── requirements.txt
├── data/
│   ├── pokemon_complete.csv   # Pokédex data (stats, types, sprites)
│   └── pokemon_types.csv      # Type effectiveness table
├── models/
│   ├── win_model.joblib       # Trained model
│   └── metrics.json           # Accuracy results
└── .streamlit/config.toml     # Dark theme
```

## Run it locally

```bash
git clone https://github.com/N1C0wastaken/poke-coach.git
cd poke-coach
pip install -r requirements.txt
streamlit run app.py
```

To retrain the model and regenerate the accuracy numbers:

```bash
python train_model.py
```

## Data sources

- Pokémon stats, types and sprite links: a Kaggle Pokédex dataset (1,350 entries, Gen 1-9), with sprites hosted in the PokéAPI sprites repository
- Type effectiveness: 18-type damage relations table

## Possible next steps

- Add real movesets and abilities (for example from PokéAPI) to make the simulator more realistic
- Multi-turn battles and switching
- Save and share team lineups

## Author

Nicolas Nunez Delgado
