
# 2 STGCN qui ont merdé ??? 

results = """
DCRNN_subway_in__e500_h4_bis1:   All Steps RMSE = 56.025, MAE = 31.694, MASE = 0.949, MAPE = 36.530
DCRNN_subway_in__e500_h4_bis2:   All Steps RMSE = 55.727, MAE = 31.392, MASE = 0.940, MAPE = 34.774
DCRNN_subway_in__e500_h4_bis3:   All Steps RMSE = 56.291, MAE = 31.756, MASE = 0.951, MAPE = 34.733
DCRNN_subway_in__e500_h4_bis4:   All Steps RMSE = 55.774, MAE = 31.195, MASE = 0.934, MAPE = 33.469
DCRNN_subway_in__e500_h4_bis5:   All Steps RMSE = 56.298, MAE = 31.793, MASE = 0.952, MAPE = 34.943

STGCN_subway_in__e500_h4_bis1:   All Steps RMSE = 49.738, MAE = 28.322, MASE = 0.848, MAPE = 31.535
STGCN_subway_in__e500_h4_bis2:   All Steps RMSE = 49.888, MAE = 28.491, MASE = 0.853, MAPE = 33.173
STGCN_subway_in__e500_h4_bis3:   All Steps RMSE = 49.170, MAE = 28.180, MASE = 0.844, MAPE = 31.499
STGCN_subway_in__e500_h4_bis4:   All Steps RMSE = 49.989, MAE = 28.551, MASE = 0.855, MAPE = 34.343
STGCN_subway_in__e500_h4_bis5:   All Steps RMSE = 50.368, MAE = 28.617, MASE = 0.857, MAPE = 32.833

STAEformer_subway_in__e200_h4_bis1:   All Steps RMSE = 46.964, MAE = 25.894, MASE = 0.775, MAPE = 27.373
STAEformer_subway_in__e200_h4_bis2:   All Steps RMSE = 47.375, MAE = 26.163, MASE = 0.783, MAPE = 28.008
STAEformer_subway_in__e200_h4_bis3:   All Steps RMSE = 47.734, MAE = 26.513, MASE = 0.794, MAPE = 28.436
STAEformer_subway_in__e200_h4_bis4:   All Steps RMSE = 47.173, MAE = 26.095, MASE = 0.781, MAPE = 28.254
STAEformer_subway_in__e200_h4_bis5:   All Steps RMSE = 47.286, MAE = 26.231, MASE = 0.785, MAPE = 28.604

DCRNN_subway_in_calendar_embedding__e500_h4_bis1:   All Steps RMSE = 50.415, MAE = 28.397, MASE = 0.850, MAPE = 32.546
DCRNN_subway_in_calendar_embedding__e500_h4_bis2:   All Steps RMSE = 50.755, MAE = 28.545, MASE = 0.855, MAPE = 32.883
DCRNN_subway_in_calendar_embedding__e500_h4_bis3:   All Steps RMSE = 51.160, MAE = 28.781, MASE = 0.862, MAPE = 32.630
DCRNN_subway_in_calendar_embedding__e500_h4_bis4:   All Steps RMSE = 51.016, MAE = 28.473, MASE = 0.852, MAPE = 32.861
DCRNN_subway_in_calendar_embedding__e500_h4_bis5:   All Steps RMSE = 51.384, MAE = 28.741, MASE = 0.860, MAPE = 31.281

STGCN_subway_in_calendar_embedding__e500_h4_bis1:   All Steps RMSE = 46.727, MAE = 26.408, MASE = 0.791, MAPE = 30.640
STGCN_subway_in_calendar_embedding__e500_h4_bis2:   All Steps RMSE = 46.702, MAE = 25.834, MASE = 0.773, MAPE = 27.333
STGCN_subway_in_calendar_embedding__e500_h4_bis3:   All Steps RMSE = 46.729, MAE = 25.800, MASE = 0.772, MAPE = 27.618
STGCN_subway_in_calendar_embedding__e500_h4_bis4:   All Steps RMSE = 48.070, MAE = 26.592, MASE = 0.796, MAPE = 28.820
STGCN_subway_in_calendar_embedding__e500_h4_bis5:   All Steps RMSE = 47.285, MAE = 26.351, MASE = 0.789, MAPE = 30.390

GMAN_subway_in_calendar__e100_h4_bis1:   All Steps RMSE = 47.743, MAE = 26.198, MASE = 0.784, MAPE = 29.863  
GMAN_subway_in_calendar__e100_h4_bis2:   All Steps RMSE = 48.272, MAE = 26.435, MASE = 0.791, MAPE = 33.647
GMAN_subway_in_calendar__e100_h4_bis3:   All Steps RMSE = 46.902, MAE = 26.235, MASE = 0.785, MAPE = 40.761
GMAN_subway_in_calendar__e100_h4_bis4:   All Steps RMSE = 48.574, MAE = 26.672, MASE = 0.799, MAPE = 34.917
GMAN_subway_in_calendar__e100_h4_bis5:   All Steps RMSE = 47.441, MAE = 26.253, MASE = 0.786, MAPE = 31.448

STAEformer_subway_in_calendar__e200_h4_bis1:   All Steps RMSE = 45.203, MAE = 24.662, MASE = 0.738, MAPE = 24.909
STAEformer_subway_in_calendar__e200_h4_bis2:   All Steps RMSE = 44.842, MAE = 24.399, MASE = 0.730, MAPE = 24.556
STAEformer_subway_in_calendar__e200_h4_bis3:   All Steps RMSE = 44.691, MAE = 24.351, MASE = 0.729, MAPE = 25.113
STAEformer_subway_in_calendar__e200_h4_bis4:   All Steps RMSE = 44.275, MAE = 24.190, MASE = 0.724, MAPE = 24.180
STAEformer_subway_in_calendar__e200_h4_bis5:   All Steps RMSE = 44.922, MAE = 24.566, MASE = 0.735, MAPE = 24.899



DCRNN_subway_in__e500_h1_bis1:   All Steps RMSE = 41.283, MAE = 24.090, MASE = 0.721, MAPE = 27.721
DCRNN_subway_in__e500_h1_bis2:   All Steps RMSE = 41.971, MAE = 24.340, MASE = 0.729, MAPE = 26.248
DCRNN_subway_in__e500_h1_bis3:   All Steps RMSE = 41.626, MAE = 24.239, MASE = 0.726, MAPE = 27.118
DCRNN_subway_in__e500_h1_bis4:   All Steps RMSE = 41.391, MAE = 24.166, MASE = 0.724, MAPE = 28.090
DCRNN_subway_in__e500_h1_bis5:   All Steps RMSE = 41.337, MAE = 24.106, MASE = 0.722, MAPE = 26.224

STGCN_subway_in__e500_h1_bis1:   All Steps RMSE = 38.624, MAE = 22.741, MASE = 0.681, MAPE = 26.307
STGCN_subway_in__e500_h1_bis2:   All Steps RMSE = 39.345, MAE = 22.863, MASE = 0.685, MAPE = 26.085
STGCN_subway_in__e500_h1_bis3:   All Steps RMSE = 38.461, MAE = 22.660, MASE = 0.678, MAPE = 25.665
STGCN_subway_in__e500_h1_bis4:   All Steps RMSE = 38.348, MAE = 22.678, MASE = 0.679, MAPE = 26.285
STGCN_subway_in__e500_h1_bis5:   All Steps RMSE = 39.122, MAE = 23.323, MASE = 0.698, MAPE = 29.498

STAEformer_subway_in__e200_h1_bis1:   All Steps RMSE = 37.357, MAE = 21.735, MASE = 0.651, MAPE = 22.944
STAEformer_subway_in__e200_h1_bis2:   All Steps RMSE = 37.324, MAE = 21.699, MASE = 0.650, MAPE = 22.909
STAEformer_subway_in__e200_h1_bis3:   All Steps RMSE = 37.255, MAE = 21.655, MASE = 0.648, MAPE = 23.118
STAEformer_subway_in__e200_h1_bis4:   All Steps RMSE = 37.062, MAE = 21.544, MASE = 0.645, MAPE = 23.009
STAEformer_subway_in__e200_h1_bis5:   All Steps RMSE = 37.418, MAE = 21.681, MASE = 0.649, MAPE = 22.702

DCRNN_subway_in_calendar_embedding__e500_h1_bis1:   All Steps RMSE = 40.397, MAE = 23.372, MASE = 0.700, MAPE = 25.541
DCRNN_subway_in_calendar_embedding__e500_h1_bis2:   All Steps RMSE = 40.519, MAE = 23.494, MASE = 0.703, MAPE = 26.602
DCRNN_subway_in_calendar_embedding__e500_h1_bis3:   All Steps RMSE = 40.832, MAE = 23.600, MASE = 0.707, MAPE = 25.950
DCRNN_subway_in_calendar_embedding__e500_h1_bis4:   All Steps RMSE = 40.366, MAE = 23.367, MASE = 0.700, MAPE = 25.692
DCRNN_subway_in_calendar_embedding__e500_h1_bis5:   All Steps RMSE = 40.566, MAE = 23.454, MASE = 0.702, MAPE = 25.493

STGCN_subway_in_calendar_embedding__e500_h1_bis1:   All Steps RMSE = 38.062, MAE = 22.416, MASE = 0.671, MAPE = 25.947
STGCN_subway_in_calendar_embedding__e500_h1_bis2:   All Steps RMSE = 38.355, MAE = 22.308, MASE = 0.668, MAPE = 23.854
STGCN_subway_in_calendar_embedding__e500_h1_bis3:   All Steps RMSE = 38.522, MAE = 22.364, MASE = 0.670, MAPE = 24.014
STGCN_subway_in_calendar_embedding__e500_h1_bis4:   All Steps RMSE = 37.855, MAE = 22.266, MASE = 0.667, MAPE = 25.361
STGCN_subway_in_calendar_embedding__e500_h1_bis5:   All Steps RMSE = 38.295, MAE = 22.458, MASE = 0.672, MAPE = 27.070

GMAN_subway_in_calendar__e100_h1_bis1:   All Steps RMSE = 39.279, MAE = 22.916, MASE = 0.686, MAPE = 27.553
GMAN_subway_in_calendar__e100_h1_bis2:   All Steps RMSE = 39.734, MAE = 23.203, MASE = 0.695, MAPE = 27.732
GMAN_subway_in_calendar__e100_h1_bis3:   All Steps RMSE = 38.815, MAE = 22.871, MASE = 0.685, MAPE = 28.935
GMAN_subway_in_calendar__e100_h1_bis4:   All Steps RMSE = 39.162, MAE = 23.323, MASE = 0.698, MAPE = 36.178
GMAN_subway_in_calendar__e100_h1_bis5:   All Steps RMSE = 39.172, MAE = 23.501, MASE = 0.704, MAPE = 33.552

STAEformer_subway_in_calendar__e200_h1_bis1:   All Steps RMSE = 36.451, MAE = 21.149, MASE = 0.633, MAPE = 21.826
STAEformer_subway_in_calendar__e200_h1_bis2:   All Steps RMSE = 36.097, MAE = 20.900, MASE = 0.626, MAPE = 21.364
STAEformer_subway_in_calendar__e200_h1_bis3:   All Steps RMSE = 36.202, MAE = 20.964, MASE = 0.628, MAPE = 21.392
STAEformer_subway_in_calendar__e200_h1_bis4:   All Steps RMSE = 36.392, MAE = 21.083, MASE = 0.631, MAPE = 21.585
STAEformer_subway_in_calendar__e200_h1_bis5:   All Steps RMSE = 36.053, MAE = 20.895, MASE = 0.626, MAPE = 21.148
"""

# --- Last run from INIT configs: 
# RNN_subway_in_calendar_embedding__e1000_h1_bis1:   All Steps RMSE = 48.205, MAE = 27.078, MASE = 0.816, MAPE = 25.569
# RNN_subway_in_calendar_embedding__e1000_h1_bis2:   All Steps RMSE = 48.106, MAE = 27.099, MASE = 0.816, MAPE = 25.933
# RNN_subway_in_calendar_embedding__e1000_h1_bis3:   All Steps RMSE = 47.308, MAE = 26.709, MASE = 0.805, MAPE = 26.128
# RNN_subway_in_calendar_embedding__e1000_h1_bis4:   All Steps RMSE = 47.329, MAE = 26.723, MASE = 0.805, MAPE = 25.690
# RNN_subway_in_calendar_embedding__e1000_h1_bis5:   All Steps RMSE = 47.878, MAE = 26.997, MASE = 0.813, MAPE = 26.076
# RNN_subway_in__e1000_h1_bis1:   All Steps RMSE = 57.054, MAE = 32.934, MASE = 0.992, MAPE = 33.960
# RNN_subway_in__e1000_h1_bis2:   All Steps RMSE = 57.006, MAE = 32.914, MASE = 0.991, MAPE = 33.727
# RNN_subway_in__e1000_h1_bis3:   All Steps RMSE = 56.974, MAE = 32.892, MASE = 0.991, MAPE = 33.834
# RNN_subway_in__e1000_h1_bis4:   All Steps RMSE = 56.969, MAE = 32.865, MASE = 0.990, MAPE = 33.294
# RNN_subway_in__e1000_h1_bis5:   All Steps RMSE = 57.037, MAE = 32.923, MASE = 0.992, MAPE = 33.783
# RNN_subway_in_calendar_embedding__e1000_h4_bis1:   All Steps RMSE = 47.002, MAE = 26.679, MASE = 0.804, MAPE = 26.290
# RNN_subway_in_calendar_embedding__e1000_h4_bis2:   All Steps RMSE = 47.469, MAE = 26.781, MASE = 0.807, MAPE = 25.462
# RNN_subway_in_calendar_embedding__e1000_h4_bis3:   All Steps RMSE = 47.312, MAE = 26.619, MASE = 0.802, MAPE = 25.756
# RNN_subway_in_calendar_embedding__e1000_h4_bis4:   All Steps RMSE = 47.495, MAE = 26.814, MASE = 0.808, MAPE = 25.540
# RNN_subway_in_calendar_embedding__e1000_h4_bis5:   All Steps RMSE = 47.353, MAE = 26.792, MASE = 0.807, MAPE = 26.522
# RNN_subway_in__e1000_h4_bis1:   All Steps RMSE = 56.818, MAE = 32.893, MASE = 0.991, MAPE = 34.089
# RNN_subway_in__e1000_h4_bis2:   All Steps RMSE = 57.094, MAE = 33.021, MASE = 0.994, MAPE = 34.207
# RNN_subway_in__e1000_h4_bis3:   All Steps RMSE = 57.482, MAE = 33.071, MASE = 0.996, MAPE = 33.754
# RNN_subway_in__e1000_h4_bis4:   All Steps RMSE = 56.910, MAE = 32.814, MASE = 0.988, MAPE = 33.558
# RNN_subway_in__e1000_h4_bis5:   All Steps RMSE = 57.021, MAE = 32.921, MASE = 0.991, MAPE = 34.024
# STAEformer_subway_in_calendar__e200_h1_bis1:   All Steps RMSE = 36.191, MAE = 20.983, MASE = 0.628, MAPE = 21.366
# STAEformer_subway_in_calendar__e200_h1_bis2:   All Steps RMSE = 36.224, MAE = 21.008, MASE = 0.629, MAPE = 21.809
# STAEformer_subway_in_calendar__e200_h1_bis3:   All Steps RMSE = 36.497, MAE = 21.160, MASE = 0.634, MAPE = 21.704
# STAEformer_subway_in_calendar__e200_h1_bis4:   All Steps RMSE = 36.200, MAE = 21.031, MASE = 0.630, MAPE = 21.354
# STAEformer_subway_in_calendar__e200_h1_bis5:   All Steps RMSE = 36.487, MAE = 21.243, MASE = 0.636, MAPE = 22.127
# STAEformer_subway_in__e200_h1_bis1:   All Steps RMSE = 36.106, MAE = 20.959, MASE = 0.628, MAPE = 21.492
# STAEformer_subway_in__e200_h1_bis2:   All Steps RMSE = 36.515, MAE = 21.168, MASE = 0.634, MAPE = 21.640
# STAEformer_subway_in__e200_h1_bis3:   All Steps RMSE = 36.428, MAE = 21.202, MASE = 0.635, MAPE = 21.975
# STAEformer_subway_in__e200_h1_bis4:   All Steps RMSE = 36.600, MAE = 21.339, MASE = 0.639, MAPE = 22.031
# STAEformer_subway_in__e200_h1_bis5:   All Steps RMSE = 36.366, MAE = 21.015, MASE = 0.629, MAPE = 21.466
# STAEformer_subway_in_calendar__e200_h4_bis1:   All Steps RMSE = 45.136, MAE = 24.394, MASE = 0.730, MAPE = 24.422
# STAEformer_subway_in_calendar__e200_h4_bis2:   All Steps RMSE = 45.006, MAE = 24.476, MASE = 0.733, MAPE = 24.716
# STAEformer_subway_in_calendar__e200_h4_bis3:   All Steps RMSE = 44.372, MAE = 24.279, MASE = 0.727, MAPE = 24.934
# STAEformer_subway_in_calendar__e200_h4_bis4:   All Steps RMSE = 44.717, MAE = 24.269, MASE = 0.727, MAPE = 24.718
# STAEformer_subway_in_calendar__e200_h4_bis5:   All Steps RMSE = 44.496, MAE = 24.364, MASE = 0.729, MAPE = 25.008
# STAEformer_subway_in__e200_h4_bis1:   All Steps RMSE = 43.945, MAE = 24.167, MASE = 0.723, MAPE = 24.470
# STAEformer_subway_in__e200_h4_bis2:   All Steps RMSE = 44.599, MAE = 24.433, MASE = 0.731, MAPE = 24.787
# STAEformer_subway_in__e200_h4_bis3:   All Steps RMSE = 44.945, MAE = 24.511, MASE = 0.734, MAPE = 24.746
# STAEformer_subway_in__e200_h4_bis4:   All Steps RMSE = 44.655, MAE = 24.404, MASE = 0.731, MAPE = 24.785
# STAEformer_subway_in__e200_h4_bis5:   All Steps RMSE = 44.345, MAE = 24.395, MASE = 0.730, MAPE = 25.070
# STGCN_subway_in_calendar_embedding__e500_h1_bis1:   All Steps RMSE = 244.696, MAE = 167.394, MASE = 5.012, MAPE = 100.000
# STGCN_subway_in_calendar_embedding__e500_h1_bis2:   All Steps RMSE = 244.696, MAE = 167.394, MASE = 5.012, MAPE = 100.000
# STGCN_subway_in_calendar_embedding__e500_h1_bis3:   All Steps RMSE = 37.855, MAE = 22.266, MASE = 0.667, MAPE = 25.361
# STGCN_subway_in_calendar_embedding__e500_h1_bis4:   All Steps RMSE = 244.696, MAE = 167.394, MASE = 5.012, MAPE = 100.000
# STGCN_subway_in_calendar_embedding__e500_h1_bis5:   All Steps RMSE = 38.236, MAE = 22.403, MASE = 0.671, MAPE = 24.924
# STGCN_subway_in__e500_h1_bis1:   All Steps RMSE = 38.624, MAE = 22.741, MASE = 0.681, MAPE = 26.307
# STGCN_subway_in__e500_h1_bis2:   All Steps RMSE = 38.844, MAE = 22.997, MASE = 0.689, MAPE = 24.527
# STGCN_subway_in__e500_h1_bis3:   All Steps RMSE = 38.348, MAE = 22.678, MASE = 0.679, MAPE = 26.285
# STGCN_subway_in__e500_h1_bis4:   All Steps RMSE = 244.696, MAE = 167.394, MASE = 5.012, MAPE = 100.000
# STGCN_subway_in__e500_h1_bis5:   All Steps RMSE = 39.122, MAE = 23.323, MASE = 0.698, MAPE = 29.498
# STGCN_subway_in_calendar_embedding__e500_h4_bis1:   All Steps RMSE = 46.589, MAE = 26.004, MASE = 0.779, MAPE = 28.283
# STGCN_subway_in_calendar_embedding__e500_h4_bis2:   All Steps RMSE = 46.727, MAE = 26.408, MASE = 0.791, MAPE = 30.640
# STGCN_subway_in_calendar_embedding__e500_h4_bis3:   All Steps RMSE = 47.285, MAE = 26.351, MASE = 0.789, MAPE = 30.390
# STGCN_subway_in_calendar_embedding__e500_h4_bis4:   All Steps RMSE = 46.490, MAE = 26.136, MASE = 0.782, MAPE = 30.568
# STGCN_subway_in_calendar_embedding__e500_h4_bis5:   All Steps RMSE = 46.048, MAE = 25.991, MASE = 0.778, MAPE = 29.859
# STGCN_subway_in__e500_h4_bis1:   All Steps RMSE = 244.630, MAE = 167.356, MASE = 5.011, MAPE = 100.000
# STGCN_subway_in__e500_h4_bis2:   All Steps RMSE = 49.888, MAE = 28.491, MASE = 0.853, MAPE = 33.173
# STGCN_subway_in__e500_h4_bis3:   All Steps RMSE = 244.630, MAE = 167.356, MASE = 5.011, MAPE = 100.000
# STGCN_subway_in__e500_h4_bis4:   All Steps RMSE = 49.989, MAE = 28.551, MASE = 0.855, MAPE = 34.343
# STGCN_subway_in__e500_h4_bis5:   All Steps RMSE = 50.368, MAE = 28.617, MASE = 0.857, MAPE = 32.833
# =======================================================


# GMAN(actualised)_subway_in_calendar__e100_h1_bis1:   All Steps RMSE = 41.374, MAE = 21.825, MASE = 0.653, MAPE = 23.616
# GMAN(actualised)_subway_in_calendar__e100_h1_bis2:   All Steps RMSE = 42.331, MAE = 22.011, MASE = 0.659, MAPE = 38.936
# GMAN(actualised)_subway_in_calendar__e100_h1_bis3:   All Steps RMSE = 40.941, MAE = 21.839, MASE = 0.654, MAPE = 30.756
# GMAN(actualised)_subway_in_calendar__e100_h1_bis4:   All Steps RMSE = 40.889, MAE = 21.742, MASE = 0.651, MAPE = 31.907
# GMAN(actualised)_subway_in_calendar__e100_h1_bis5:   All Steps RMSE = 39.658, MAE = 21.865, MASE = 0.655, MAPE = 23.875


# GMAN(actualised)_subway_in_calendar__e100_h4_bis1:   All Steps RMSE = 47.567, MAE = 24.577, MASE = 0.736, MAPE = 37.698
# GMAN(actualised)_subway_in_calendar__e100_h4_bis2:   All Steps RMSE = 49.792, MAE = 25.278, MASE = 0.757, MAPE = 31.262
# GMAN(actualised)_subway_in_calendar__e100_h4_bis3:   All Steps RMSE = 49.005, MAE = 25.027, MASE = 0.749, MAPE = 34.472
# GMAN(actualised)_subway_in_calendar__e100_h4_bis4:   All Steps RMSE = 50.418, MAE = 25.146, MASE = 0.753, MAPE = 26.479
# GMAN(actualised)_subway_in_calendar__e100_h4_bis5:   All Steps RMSE = 47.948, MAE = 25.057, MASE = 0.750, MAPE = 30.462