# --------- load_config.py ---------
import argparse

parser = argparse.ArgumentParser(description='SARIMAX')

# Arguments ARIMA
parser.add_argument('--order', type=int, nargs=3, default=[1, 0, 0],
                    help="Ordre ARIMA (p d q)")
parser.add_argument('--seasonal_order', type=int, nargs=4, default=[0, 0, 0, 0],
                    help="Ordre saisonnier ARIMA (P D Q s)")
parser.add_argument('--enforce_stationarity', type=bool, default=True,
                    help="Faire stationnarité")
parser.add_argument('--enforce_invertibility', type=bool, default=True,
                    help="Faire inversionnable")

args = parser.parse_args(args=[])