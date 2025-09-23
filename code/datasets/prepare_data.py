import os
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    raw_dir = project_root / 'data' / 'raw'
    processed_dir = project_root / 'data' / 'processed'

    movies = pd.read_csv(raw_dir / 'movies.csv')
    directors = pd.read_csv(raw_dir / 'directors.csv')

    df = movies.merge(
        directors,
        left_on='director_id',
        right_on='id',
        suffixes=('_movie', '_director'),
    )

    features = (
        df.groupby('director_name')
        .agg({
            'budget': 'mean',
            'revenue': 'mean',
            'vote_average': 'mean',
            'vote_count': 'mean',
            'popularity': 'mean',
            'title': 'count',
        })
        .reset_index()
    )

    features.rename(
        columns={
            'budget': 'avg_budget',
            'revenue': 'avg_revenue',
            'vote_average': 'avg_vote',
            'vote_count': 'avg_vote_count',
            'popularity': 'avg_popularity',
            'title': 'num_movies',
        },
        inplace=True,
    )

    model_cols = [
        'avg_budget',
        'avg_revenue',
        'avg_vote',
        'avg_vote_count',
        'avg_popularity',
        'num_movies',
    ]
    features = features.dropna(subset=model_cols)

    features['funding_class'] = pd.qcut(
        features['avg_revenue'], 5, labels=[0, 1, 2, 3, 4]
    )

    processed_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df = train_test_split(features, test_size=0.2, random_state=42)
    train_df.to_csv(processed_dir / 'train.csv', index=False)
    test_df.to_csv(processed_dir / 'test.csv', index=False)
    features.to_csv(processed_dir / 'directors_features.csv', index=False)

    print('Saved: data/processed/train.csv, test.csv, directors_features.csv')


if __name__ == '__main__':
    main()
