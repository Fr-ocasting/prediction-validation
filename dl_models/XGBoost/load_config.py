# --------- load_config.py ---------
import argparse

parser = argparse.ArgumentParser(description='XgBoost')

parser.add_argument('--n_estimators', type=int, default=100,
                    help="Nombre d'arbres")
parser.add_argument('--max_depth', type=int, default=6,
                    help="Profondeur maximale des arbres")
parser.add_argument('--subsample', type=float, default=1.0,
                    help="Ratio d'échantillonnage des instances")
parser.add_argument('--colsample_bytree', type=float, default=1.0,
                    help="Ratio d'échantillonnage des colonnes par arbre")
parser.add_argument('--gamma', type=float, default=0.0,
                    help="Gain minimal de réduction de perte pour un split")
# parser.add_argument('--reg_alpha', type=float, default=0.0,
#                     help="L1 regularization")
# parser.add_argument('--reg_lambda', type=float, default=1.0,
#                     help="L2 regularization")
# parser.add_argument('--objective', type=str, default='reg:squarederror',
#                     help="Objectif d'apprentissage")
# parser.add_argument('--eval_metric', type=str, default='rmse',
#                     help="Métrique d'évaluation")

args = parser.parse_args(args=[])