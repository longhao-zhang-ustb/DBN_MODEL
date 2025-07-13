def get_lgb_param():
    params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_boost_round': 200,
        'learning_rate': 0.1,
        'verbose': 5,
        'n_estimators': 150,
        'num_iterations': 500
    }
    return params
