from ray import tune

config = {
    'n_estimators': tune.choice([50, 100, 200, 500]),
    'learning_rate': tune.loguniform(1e-4, 1e-1),
    'max_depth': tune.choice([3, 6, 9, 12]),
    'subsample': tune.uniform(0.5, 1.0),
    'colsample_bytree': tune.uniform(0.5, 1.0),
    'gamma': tune.uniform(0.0, 5.0),
    'reg_alpha': tune.loguniform(1e-3, 1e1),
    'reg_lambda': tune.loguniform(1e-3, 1e1),
}