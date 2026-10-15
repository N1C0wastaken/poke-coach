import json
import joblib
import numpy as np
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from battle import load_battle_data, simulate, build_features, best_stab

N_PAIRS = 300_000
rng = np.random.default_rng(42)

df, arr = load_battle_data()
n = len(df)

# 1. simulate random matchups
i = rng.integers(0, n, N_PAIRS)
j = rng.integers(0, n, N_PAIRS)
keep = i != j
i, j = i[keep], j[keep]
win, draw = simulate(arr, i, j, rng)
i, j, y = i[~draw], j[~draw], win[~draw]
X = build_features(arr, i, j)

# 2. hold out 20% of SPECIES: test pairs use only species the model never saw
held_out = np.zeros(n, dtype=bool)
held_out[rng.permutation(n)[: int(0.2 * n)]] = True
train_mask = ~held_out[i] & ~held_out[j]
test_mask = held_out[i] & held_out[j]

# 3. train two models
logreg = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
logreg.fit(X[train_mask], y[train_mask])
gboost = HistGradientBoostingClassifier(random_state=0)
gboost.fit(X[train_mask], y[train_mask])


# 4. baseline: does the hand-built formula pick the simulated winner?
def formula_score(a, b, w=(3, 3, 1, 1, 1)):
    s = arr["stats"]
    lg = lambda m: np.log2(np.maximum(m, 0.25))
    bulk = (s["hp"][a] * s["defense"][a] * s["sp_defense"][a]) ** (1 / 3) / 100
    power = np.maximum(s["attack"][a], s["sp_attack"][a]) / 100
    return (w[0] * lg(best_stab(arr, a, b)) - w[1] * lg(best_stab(arr, b, a))
            + w[2] * (s["speed"][a] > s["speed"][b]) + w[3] * bulk + w[4] * power)


formula_pred = formula_score(i, j) > formula_score(j, i)

metrics = {
    "train_pairs": int(train_mask.sum()),
    "test_pairs": int(test_mask.sum()),
    "formula_accuracy": float((formula_pred[test_mask] == (y[test_mask] == 1)).mean()),
    "logistic_accuracy": float(logreg.score(X[test_mask], y[test_mask])),
    "gradient_boost_accuracy": float(gboost.score(X[test_mask], y[test_mask])),
}
print(json.dumps(metrics, indent=2))

Path("models").mkdir(exist_ok=True)
joblib.dump(gboost, "models/win_model.joblib")
Path("models/metrics.json").write_text(json.dumps(metrics, indent=2))