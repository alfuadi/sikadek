from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

def train_models(df):
    X = df.drop(columns=['label'])
    y = df['label']
    rf = RandomForestClassifier().fit(X, y)
    mlp = MLPClassifier(max_iter=500).fit(X, y)
    return rf, mlp

def evaluate_models(models, X, y):
    rf, mlp = models
    metrics = {}
    for name, model in zip(['RF', 'MLP'], [rf, mlp]):
        prob = model.predict_proba(X)[:, 1]
        metrics[name] = {
            'roc_auc': roc_auc_score(y, prob),
            'report': classification_report(y, model.predict(X), output_dict=True)
        }
    return metrics
