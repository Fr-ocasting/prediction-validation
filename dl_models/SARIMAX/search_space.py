from ray import tune

config = {
    'order': tune.choice([[1, 0, 0], [2, 1, 0], [1, 1, 1]]),
    'seasonal_order': tune.choice([[0, 0, 0, 0], [1, 0, 1, 7], [1, 1, 1, 12]]),
    'enforce_stationarity': tune.choice([True, False]),
    'enforce_invertibility': tune.choice([True, False]),
}