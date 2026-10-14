"""Opt-in descriptive analysis of *user-supplied* original Gaze3P ratings.

This script DOES NOT download the full dataset. Use it only after checking the
Gaze3P dataset license and manually supplying a properly formed CSV extract.
Requires a column containing integer privacy ratings in [1,7]. Other columns
are never presumed to be in the published data layout.

Example:
  python src/dataset_ratings_summary.py --input path/to/extract.csv --rating-col privacy_rating
  python src/dataset_ratings_summary.py --input extract.csv --rating-col rating --group-col attribute
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def run(inp: Path, rating_col: str, group_col: str | None, outdir: Path):
    df = pd.read_csv(inp)
    if rating_col not in df.columns:
        raise ValueError(f'Rating column {rating_col!r} missing; available columns: {list(df.columns)}')
    if group_col and group_col not in df.columns:
        raise ValueError(f'Group column {group_col!r} missing; available columns: {list(df.columns)}')
    rating = pd.to_numeric(df[rating_col], errors='coerce')
    valid = rating.notna() & rating.between(1, 7) & (rating % 1 == 0)
    cleaned = df.loc[valid].copy()
    cleaned[rating_col] = rating.loc[valid].astype(int)
    outdir.mkdir(parents=True, exist_ok=True)
    totals = cleaned[rating_col].value_counts().reindex(range(1, 8), fill_value=0)
    pd.DataFrame({'rating': totals.index, 'count': totals.values,
                  'proportion': totals.values / max(len(cleaned), 1)}).to_csv(
        outdir / 'real_dataset_rating_counts.csv', index=False)
    if group_col:
        group = cleaned.groupby(group_col)[rating_col].agg(['count', 'mean', 'median']).reset_index()
        group.to_csv(outdir / 'real_dataset_group_summary.csv', index=False)
    print(f'Read {len(df)} rows; {len(cleaned)} valid ratings; {len(df)-len(cleaned)} skipped.')
    print(totals.to_string())
    if len(cleaned) == 0:
        raise ValueError('No valid rows; inspect input CSV and selected column.')

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path)
    p.add_argument('--rating-col', required=True)
    p.add_argument('--group-col', default=None)
    p.add_argument('--output-dir', type=Path, default=Path('outputs/real_dataset'))
    a = p.parse_args()
    run(a.input, a.rating_col, a.group_col, a.output_dir)
