import pandas as pd
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    processed_dir = project_root / 'data' / 'processed'
    models_dir = project_root / 'models'

    train_df = pd.read_csv(processed_dir / 'train.csv')
    test_df = pd.read_csv(processed_dir / 'test.csv')

    feature_cols = [
        'avg_budget',
        'avg_revenue',
        'avg_vote',
        'num_movies',
        'avg_popularity',
        'avg_vote_count',
    ]

    X_train = train_df[feature_cols]
    y_train = train_df['funding_class']
    X_test = test_df[feature_cols]
    y_test = test_df['funding_class']

    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    models_dir.mkdir(parents=True, exist_ok=True)
    with open(models_dir / 'director_model.pkl', 'wb') as f:
        pickle.dump(clf, f)

    print('Saved to models/director_model.pkl')


if __name__ == '__main__':
    main()
