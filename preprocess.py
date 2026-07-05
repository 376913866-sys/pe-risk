import numpy as np
import pandas as pd

def calc_hsi(ast, alt, bmi):
    try:
        if ast <= 0 or alt <= 0:
            return 0.0
        return 8 * (alt / ast) + bmi + 2
    except:
        return 0.0


def safe_dataframe(df, features):
    df = df.reindex(columns=features)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.mean())
    return df.astype(np.float32)


def format_ga(x):
    try:
        w = int(x)
        d = round((x - w) * 7)
        return f"{w}+{d}"
    except:
        return "N/A"
