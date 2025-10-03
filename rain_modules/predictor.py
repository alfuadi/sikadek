def predict(models, latest_data):
    rf, mlp = models
    prob_rf = rf.predict_proba(latest_data)[:, 1][0]
    prob_mlp = mlp.predict_proba(latest_data)[:, 1][0]
    return prob_rf * 100, prob_mlp * 100
