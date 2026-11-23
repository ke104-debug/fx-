import lightgbm as lgb
import pandas as pd

from .features import feature_cols

PARAMS = dict(n_estimators=300, learning_rate=0.02, num_leaves=15, min_child_samples=200,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0,
              verbose=-1, random_state=42)


def train(f: pd.DataFrame) -> lgb.LGBMClassifier:
    d = f.dropna(subset=["target"])
    cols = feature_cols(d)
    m = lgb.LGBMClassifier(**PARAMS)
    m.fit(d[cols], d["target"].astype(int), categorical_feature=["pair"])
    return m


def predict_proba(m, f: pd.DataFrame) -> pd.Series:
    return pd.Series(m.predict_proba(f[feature_cols(f)])[:, 1], index=f.index)
