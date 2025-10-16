# # Trials done: 
# Prediction of Bike-out :
# ----- 
# Bike_out    + ( calendar late fusion )
# Bike_out    + ( calendar late fusion )   +   ( Weather early fusion repeat_t_proj )


# Prediction of Subway-out : 
# -----
# Subway_out
# Subway_out  + ( calendar late fusion )   +   ( Subway_in shared_embedding ) 
# Subway_out  + ( calendar late fusion )   +   ( Weather early fusion repeat_t_proj ) +  ( Subway_in shared_embedding ) 



results = """
STAEformer_bike_out_5p__e50_h1_bis1:   All Steps RMSE = 4.801, MAE = 3.081, MASE = 0.800, MAPE = 52.614
STAEformer_bike_out_5p__e50_h1_bis2:   All Steps RMSE = 4.765, MAE = 3.034, MASE = 0.788, MAPE = 49.289
STAEformer_bike_out_5p__e50_h1_bis3:   All Steps RMSE = 4.725, MAE = 3.015, MASE = 0.783, MAPE = 49.737
STAEformer_bike_out_5p__e50_h1_bis4:   All Steps RMSE = 4.720, MAE = 3.042, MASE = 0.790, MAPE = 51.734
STAEformer_bike_out_5p__e50_h1_bis5:   All Steps RMSE = 4.816, MAE = 3.077, MASE = 0.799, MAPE = 52.489
STAEformer_bike_out_10p__e50_h1_bis1:   All Steps RMSE = 4.633, MAE = 2.981, MASE = 0.774, MAPE = 50.206
STAEformer_bike_out_10p__e50_h1_bis2:   All Steps RMSE = 4.630, MAE = 2.980, MASE = 0.774, MAPE = 50.512
STAEformer_bike_out_10p__e50_h1_bis3:   All Steps RMSE = 4.549, MAE = 2.937, MASE = 0.763, MAPE = 49.293
STAEformer_bike_out_10p__e50_h1_bis4:   All Steps RMSE = 4.635, MAE = 2.967, MASE = 0.770, MAPE = 48.754
STAEformer_bike_out_10p__e50_h1_bis5:   All Steps RMSE = 4.666, MAE = 2.984, MASE = 0.775, MAPE = 49.349
STAEformer_bike_out_15p__e50_h1_bis1:   All Steps RMSE = 4.582, MAE = 2.957, MASE = 0.768, MAPE = 49.474
STAEformer_bike_out_15p__e50_h1_bis2:   All Steps RMSE = 4.567, MAE = 2.920, MASE = 0.758, MAPE = 46.846
STAEformer_bike_out_15p__e50_h1_bis3:   All Steps RMSE = 4.553, MAE = 2.914, MASE = 0.757, MAPE = 47.375
STAEformer_bike_out_15p__e50_h1_bis4:   All Steps RMSE = 4.534, MAE = 2.918, MASE = 0.758, MAPE = 49.102
STAEformer_bike_out_15p__e50_h1_bis5:   All Steps RMSE = 4.584, MAE = 2.945, MASE = 0.765, MAPE = 48.836
STAEformer_bike_out_25p__e50_h1_bis1:   All Steps RMSE = 4.467, MAE = 2.870, MASE = 0.745, MAPE = 45.100
STAEformer_bike_out_25p__e50_h1_bis2:   All Steps RMSE = 4.447, MAE = 2.886, MASE = 0.749, MAPE = 48.621
STAEformer_bike_out_25p__e50_h1_bis3:   All Steps RMSE = 4.466, MAE = 2.914, MASE = 0.757, MAPE = 49.674
STAEformer_bike_out_25p__e50_h1_bis4:   All Steps RMSE = 4.451, MAE = 2.875, MASE = 0.747, MAPE = 48.158
STAEformer_bike_out_25p__e50_h1_bis5:   All Steps RMSE = 4.520, MAE = 2.901, MASE = 0.753, MAPE = 47.257
STAEformer_bike_out_50p__e50_h1_bis1:   All Steps RMSE = 4.419, MAE = 2.852, MASE = 0.740, MAPE = 46.463
STAEformer_bike_out_50p__e50_h1_bis2:   All Steps RMSE = 4.357, MAE = 2.817, MASE = 0.731, MAPE = 45.129
STAEformer_bike_out_50p__e50_h1_bis3:   All Steps RMSE = 4.396, MAE = 2.829, MASE = 0.735, MAPE = 45.497
STAEformer_bike_out_50p__e50_h1_bis4:   All Steps RMSE = 4.458, MAE = 2.881, MASE = 0.748, MAPE = 46.644
STAEformer_bike_out_50p__e50_h1_bis5:   All Steps RMSE = 4.367, MAE = 2.816, MASE = 0.731, MAPE = 45.079
STAEformer_bike_out_75p__e50_h1_bis1:   All Steps RMSE = 4.371, MAE = 2.848, MASE = 0.740, MAPE = 48.375
STAEformer_bike_out_75p__e50_h1_bis2:   All Steps RMSE = 4.398, MAE = 2.835, MASE = 0.736, MAPE = 45.566
STAEformer_bike_out_75p__e50_h1_bis3:   All Steps RMSE = 4.312, MAE = 2.828, MASE = 0.734, MAPE = 48.956
STAEformer_bike_out_75p__e50_h1_bis4:   All Steps RMSE = 4.336, MAE = 2.793, MASE = 0.725, MAPE = 44.648
STAEformer_bike_out_75p__e50_h1_bis5:   All Steps RMSE = 4.322, MAE = 2.806, MASE = 0.729, MAPE = 47.058
STAEformer_bike_out_100p__e50_h1_bis1:   All Steps RMSE = 4.278, MAE = 2.765, MASE = 0.718, MAPE = 45.584
STAEformer_bike_out_100p__e50_h1_bis2:   All Steps RMSE = 4.302, MAE = 2.782, MASE = 0.722, MAPE = 45.577
STAEformer_bike_out_100p__e50_h1_bis3:   All Steps RMSE = 4.322, MAE = 2.790, MASE = 0.724, MAPE = 46.183
STAEformer_bike_out_100p__e50_h1_bis4:   All Steps RMSE = 4.317, MAE = 2.785, MASE = 0.723, MAPE = 45.711
STAEformer_bike_out_100p__e50_h1_bis5:   All Steps RMSE = 4.293, MAE = 2.778, MASE = 0.721, MAPE = 45.983

STAEformer_bike_out_5p__e50_h4_bis1:   All Steps RMSE = 5.526, MAE = 3.447, MASE = 0.895, MAPE = 57.014
STAEformer_bike_out_5p__e50_h4_bis2:   All Steps RMSE = 5.510, MAE = 3.480, MASE = 0.904, MAPE = 61.117
STAEformer_bike_out_5p__e50_h4_bis3:   All Steps RMSE = 5.607, MAE = 3.487, MASE = 0.905, MAPE = 57.894
STAEformer_bike_out_5p__e50_h4_bis4:   All Steps RMSE = 5.591, MAE = 3.466, MASE = 0.900, MAPE = 56.046
STAEformer_bike_out_5p__e50_h4_bis5:   All Steps RMSE = 5.656, MAE = 3.498, MASE = 0.908, MAPE = 58.353
STAEformer_bike_out_10p__e50_h4_bis1:   All Steps RMSE = 5.342, MAE = 3.336, MASE = 0.866, MAPE = 54.716
STAEformer_bike_out_10p__e50_h4_bis2:   All Steps RMSE = 5.367, MAE = 3.352, MASE = 0.870, MAPE = 55.781
STAEformer_bike_out_10p__e50_h4_bis3:   All Steps RMSE = 5.453, MAE = 3.408, MASE = 0.885, MAPE = 57.497
STAEformer_bike_out_10p__e50_h4_bis4:   All Steps RMSE = 5.544, MAE = 3.454, MASE = 0.897, MAPE = 59.262
STAEformer_bike_out_10p__e50_h4_bis5:   All Steps RMSE = 5.552, MAE = 3.432, MASE = 0.891, MAPE = 56.380
STAEformer_bike_out_15p__e50_h4_bis1:   All Steps RMSE = 5.308, MAE = 3.323, MASE = 0.863, MAPE = 56.401
STAEformer_bike_out_15p__e50_h4_bis2:   All Steps RMSE = 5.323, MAE = 3.308, MASE = 0.859, MAPE = 54.838
STAEformer_bike_out_15p__e50_h4_bis3:   All Steps RMSE = 5.223, MAE = 3.315, MASE = 0.861, MAPE = 58.430
STAEformer_bike_out_15p__e50_h4_bis4:   All Steps RMSE = 5.341, MAE = 3.312, MASE = 0.860, MAPE = 54.215
STAEformer_bike_out_15p__e50_h4_bis5:   All Steps RMSE = 5.362, MAE = 3.348, MASE = 0.869, MAPE = 57.283
STAEformer_bike_out_25p__e50_h4_bis1:   All Steps RMSE = 5.040, MAE = 3.182, MASE = 0.826, MAPE = 53.205
STAEformer_bike_out_25p__e50_h4_bis2:   All Steps RMSE = 5.087, MAE = 3.218, MASE = 0.835, MAPE = 54.185
STAEformer_bike_out_25p__e50_h4_bis3:   All Steps RMSE = 5.160, MAE = 3.230, MASE = 0.839, MAPE = 52.425
STAEformer_bike_out_25p__e50_h4_bis4:   All Steps RMSE = 5.122, MAE = 3.184, MASE = 0.827, MAPE = 49.098
STAEformer_bike_out_25p__e50_h4_bis5:   All Steps RMSE = 5.160, MAE = 3.241, MASE = 0.841, MAPE = 55.017
STAEformer_bike_out_50p__e50_h4_bis1:   All Steps RMSE = 4.998, MAE = 3.126, MASE = 0.812, MAPE = 50.411
STAEformer_bike_out_50p__e50_h4_bis2:   All Steps RMSE = 5.174, MAE = 3.221, MASE = 0.836, MAPE = 53.232
STAEformer_bike_out_50p__e50_h4_bis3:   All Steps RMSE = 4.937, MAE = 3.102, MASE = 0.805, MAPE = 51.290
STAEformer_bike_out_50p__e50_h4_bis4:   All Steps RMSE = 4.986, MAE = 3.112, MASE = 0.808, MAPE = 50.124
STAEformer_bike_out_50p__e50_h4_bis5:   All Steps RMSE = 4.965, MAE = 3.105, MASE = 0.806, MAPE = 49.747
STAEformer_bike_out_75p__e50_h4_bis1:   All Steps RMSE = 4.947, MAE = 3.091, MASE = 0.803, MAPE = 50.226
STAEformer_bike_out_75p__e50_h4_bis2:   All Steps RMSE = 4.919, MAE = 3.075, MASE = 0.798, MAPE = 49.651
STAEformer_bike_out_75p__e50_h4_bis3:   All Steps RMSE = 5.109, MAE = 3.174, MASE = 0.824, MAPE = 51.570
STAEformer_bike_out_75p__e50_h4_bis4:   All Steps RMSE = 4.957, MAE = 3.133, MASE = 0.813, MAPE = 53.365
STAEformer_bike_out_75p__e50_h4_bis5:   All Steps RMSE = 4.882, MAE = 3.065, MASE = 0.796, MAPE = 49.854
STAEformer_bike_out_100p__e50_h4_bis1:   All Steps RMSE = 4.818, MAE = 3.030, MASE = 0.787, MAPE = 50.193
STAEformer_bike_out_100p__e50_h4_bis2:   All Steps RMSE = 4.810, MAE = 3.042, MASE = 0.790, MAPE = 50.630
STAEformer_bike_out_100p__e50_h4_bis3:   All Steps RMSE = 4.837, MAE = 3.039, MASE = 0.789, MAPE = 49.305
STAEformer_bike_out_100p__e50_h4_bis4:   All Steps RMSE = 4.963, MAE = 3.087, MASE = 0.801, MAPE = 49.814
STAEformer_bike_out_100p__e50_h4_bis5:   All Steps RMSE = 4.862, MAE = 3.051, MASE = 0.792, MAPE = 49.790

STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis1:   All Steps RMSE = 5.811, MAE = 3.558, MASE = 0.924, MAPE = 56.862
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis2:   All Steps RMSE = 5.700, MAE = 3.608, MASE = 0.937, MAPE = 64.701
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis3:   All Steps RMSE = 5.554, MAE = 3.466, MASE = 0.900, MAPE = 58.895
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis4:   All Steps RMSE = 5.643, MAE = 3.470, MASE = 0.901, MAPE = 54.356
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis5:   All Steps RMSE = 5.847, MAE = 3.679, MASE = 0.955, MAPE = 65.465
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis1:   All Steps RMSE = 5.417, MAE = 3.366, MASE = 0.874, MAPE = 56.029
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis2:   All Steps RMSE = 5.334, MAE = 3.344, MASE = 0.868, MAPE = 56.829
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis3:   All Steps RMSE = 5.321, MAE = 3.330, MASE = 0.865, MAPE = 54.706
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis4:   All Steps RMSE = 5.737, MAE = 3.655, MASE = 0.949, MAPE = 67.448
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis5:   All Steps RMSE = 5.281, MAE = 3.315, MASE = 0.861, MAPE = 55.416
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis1:   All Steps RMSE = 5.229, MAE = 3.273, MASE = 0.850, MAPE = 55.213
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis2:   All Steps RMSE = 5.326, MAE = 3.324, MASE = 0.863, MAPE = 55.404
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis3:   All Steps RMSE = 5.222, MAE = 3.302, MASE = 0.857, MAPE = 56.377
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis4:   All Steps RMSE = 5.197, MAE = 3.253, MASE = 0.845, MAPE = 53.567
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis5:   All Steps RMSE = 5.283, MAE = 3.318, MASE = 0.862, MAPE = 55.993
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis1:   All Steps RMSE = 5.050, MAE = 3.167, MASE = 0.822, MAPE = 51.616
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis2:   All Steps RMSE = 5.070, MAE = 3.172, MASE = 0.824, MAPE = 50.967
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis3:   All Steps RMSE = 5.054, MAE = 3.180, MASE = 0.826, MAPE = 52.430
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis4:   All Steps RMSE = 5.097, MAE = 3.180, MASE = 0.826, MAPE = 49.491
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis5:   All Steps RMSE = 5.166, MAE = 3.226, MASE = 0.838, MAPE = 52.355
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis1:   All Steps RMSE = 5.049, MAE = 3.163, MASE = 0.821, MAPE = 49.980
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis2:   All Steps RMSE = 4.918, MAE = 3.111, MASE = 0.808, MAPE = 51.448
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis3:   All Steps RMSE = 5.041, MAE = 3.162, MASE = 0.821, MAPE = 51.931
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis4:   All Steps RMSE = 4.946, MAE = 3.114, MASE = 0.809, MAPE = 51.303
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis5:   All Steps RMSE = 5.024, MAE = 3.146, MASE = 0.817, MAPE = 49.877
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis1:   All Steps RMSE = 5.021, MAE = 3.137, MASE = 0.815, MAPE = 50.320
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis2:   All Steps RMSE = 4.927, MAE = 3.108, MASE = 0.807, MAPE = 52.194
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis3:   All Steps RMSE = 4.910, MAE = 3.119, MASE = 0.810, MAPE = 53.625
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis4:   All Steps RMSE = 4.929, MAE = 3.081, MASE = 0.800, MAPE = 48.751
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis5:   All Steps RMSE = 4.954, MAE = 3.108, MASE = 0.807, MAPE = 49.809
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis1:   All Steps RMSE = 4.876, MAE = 3.096, MASE = 0.804, MAPE = 52.756
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis2:   All Steps RMSE = 4.839, MAE = 3.055, MASE = 0.793, MAPE = 50.005
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis3:   All Steps RMSE = 4.795, MAE = 3.056, MASE = 0.793, MAPE = 51.834
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis4:   All Steps RMSE = 4.865, MAE = 3.055, MASE = 0.793, MAPE = 49.737
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis5:   All Steps RMSE = 4.866, MAE = 3.052, MASE = 0.792, MAPE = 49.405

STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis1:   All Steps RMSE = 4.721, MAE = 3.067, MASE = 0.796, MAPE = 53.343
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis2:   All Steps RMSE = 4.670, MAE = 3.001, MASE = 0.779, MAPE = 49.107
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis3:   All Steps RMSE = 4.750, MAE = 3.037, MASE = 0.789, MAPE = 49.346
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis4:   All Steps RMSE = 4.713, MAE = 2.992, MASE = 0.777, MAPE = 47.548
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis5:   All Steps RMSE = 4.643, MAE = 3.002, MASE = 0.779, MAPE = 50.396
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis1:   All Steps RMSE = 4.582, MAE = 2.953, MASE = 0.767, MAPE = 48.802
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis2:   All Steps RMSE = 4.565, MAE = 2.947, MASE = 0.765, MAPE = 49.356
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis3:   All Steps RMSE = 4.564, MAE = 2.962, MASE = 0.769, MAPE = 50.289
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis4:   All Steps RMSE = 4.658, MAE = 2.979, MASE = 0.773, MAPE = 48.624
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis5:   All Steps RMSE = 4.613, MAE = 2.985, MASE = 0.775, MAPE = 50.792
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis1:   All Steps RMSE = 4.585, MAE = 2.952, MASE = 0.767, MAPE = 49.363
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis2:   All Steps RMSE = 4.521, MAE = 2.936, MASE = 0.762, MAPE = 49.987
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis3:   All Steps RMSE = 4.533, MAE = 2.911, MASE = 0.756, MAPE = 47.641
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis4:   All Steps RMSE = 4.561, MAE = 2.930, MASE = 0.761, MAPE = 47.752
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis5:   All Steps RMSE = 4.547, MAE = 2.926, MASE = 0.760, MAPE = 48.320
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis1:   All Steps RMSE = 4.479, MAE = 2.876, MASE = 0.747, MAPE = 46.588
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis2:   All Steps RMSE = 4.459, MAE = 2.879, MASE = 0.748, MAPE = 47.037
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis3:   All Steps RMSE = 4.492, MAE = 2.896, MASE = 0.752, MAPE = 46.752
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis4:   All Steps RMSE = 4.463, MAE = 2.891, MASE = 0.751, MAPE = 48.190
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis5:   All Steps RMSE = 4.441, MAE = 2.859, MASE = 0.742, MAPE = 46.695
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis1:   All Steps RMSE = 4.340, MAE = 2.817, MASE = 0.732, MAPE = 46.035
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis2:   All Steps RMSE = 4.344, MAE = 2.828, MASE = 0.734, MAPE = 46.552
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis3:   All Steps RMSE = 4.393, MAE = 2.830, MASE = 0.735, MAPE = 45.159
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis4:   All Steps RMSE = 4.448, MAE = 2.869, MASE = 0.745, MAPE = 46.895
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis5:   All Steps RMSE = 4.345, MAE = 2.820, MASE = 0.732, MAPE = 45.582
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis1:   All Steps RMSE = 4.378, MAE = 2.827, MASE = 0.734, MAPE = 44.840
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis2:   All Steps RMSE = 4.347, MAE = 2.809, MASE = 0.729, MAPE = 45.828
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis3:   All Steps RMSE = 4.313, MAE = 2.790, MASE = 0.725, MAPE = 45.438
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis4:   All Steps RMSE = 4.335, MAE = 2.794, MASE = 0.726, MAPE = 43.624
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis5:   All Steps RMSE = 4.339, MAE = 2.829, MASE = 0.735, MAPE = 47.513
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h1_bis1:   All Steps RMSE = 4.274, MAE = 2.795, MASE = 0.726, MAPE = 47.085
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h1_bis2:   All Steps RMSE = 4.286, MAE = 2.767, MASE = 0.718, MAPE = 44.595
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h1_bis3:   All Steps RMSE = 4.276, MAE = 2.787, MASE = 0.724, MAPE = 47.019
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h1_bis4:   All Steps RMSE = 4.282, MAE = 2.767, MASE = 0.719, MAPE = 45.458
STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h1_bis5:   All Steps RMSE = 4.317, MAE = 2.792, MASE = 0.725, MAPE = 45.808




STAEformer_subway_out_5p__e50_h1_bis1:   All Steps RMSE = 43.131, MAE = 24.867, MASE = 0.770, MAPE = 24.611
STAEformer_subway_out_5p__e50_h1_bis2:   All Steps RMSE = 42.660, MAE = 24.251, MASE = 0.751, MAPE = 25.365
STAEformer_subway_out_5p__e50_h1_bis3:   All Steps RMSE = 42.381, MAE = 24.290, MASE = 0.752, MAPE = 26.485
STAEformer_subway_out_5p__e50_h1_bis4:   All Steps RMSE = 42.311, MAE = 24.347, MASE = 0.754, MAPE = 25.525
STAEformer_subway_out_5p__e50_h1_bis5:   All Steps RMSE = 42.830, MAE = 24.674, MASE = 0.764, MAPE = 26.180
STAEformer_subway_out_10p__e50_h1_bis1:   All Steps RMSE = 41.991, MAE = 24.011, MASE = 0.744, MAPE = 24.643
STAEformer_subway_out_10p__e50_h1_bis2:   All Steps RMSE = 41.678, MAE = 23.865, MASE = 0.739, MAPE = 24.928
STAEformer_subway_out_10p__e50_h1_bis3:   All Steps RMSE = 42.063, MAE = 23.975, MASE = 0.743, MAPE = 25.543
STAEformer_subway_out_10p__e50_h1_bis4:   All Steps RMSE = 41.450, MAE = 23.703, MASE = 0.734, MAPE = 25.993
STAEformer_subway_out_10p__e50_h1_bis5:   All Steps RMSE = 41.462, MAE = 23.731, MASE = 0.735, MAPE = 25.434
STAEformer_subway_out_15p__e50_h1_bis1:   All Steps RMSE = 40.604, MAE = 23.208, MASE = 0.719, MAPE = 24.961
STAEformer_subway_out_15p__e50_h1_bis2:   All Steps RMSE = 41.281, MAE = 23.515, MASE = 0.728, MAPE = 24.153
STAEformer_subway_out_15p__e50_h1_bis3:   All Steps RMSE = 41.137, MAE = 23.464, MASE = 0.727, MAPE = 23.887
STAEformer_subway_out_15p__e50_h1_bis4:   All Steps RMSE = 40.846, MAE = 23.351, MASE = 0.723, MAPE = 24.768
STAEformer_subway_out_15p__e50_h1_bis5:   All Steps RMSE = 40.937, MAE = 23.272, MASE = 0.721, MAPE = 24.620
STAEformer_subway_out_25p__e50_h1_bis1:   All Steps RMSE = 39.880, MAE = 22.424, MASE = 0.695, MAPE = 23.827
STAEformer_subway_out_25p__e50_h1_bis2:   All Steps RMSE = 39.760, MAE = 22.527, MASE = 0.698, MAPE = 24.010
STAEformer_subway_out_25p__e50_h1_bis3:   All Steps RMSE = 39.690, MAE = 22.436, MASE = 0.695, MAPE = 22.835
STAEformer_subway_out_25p__e50_h1_bis4:   All Steps RMSE = 39.662, MAE = 22.404, MASE = 0.694, MAPE = 23.182
STAEformer_subway_out_25p__e50_h1_bis5:   All Steps RMSE = 39.979, MAE = 22.675, MASE = 0.702, MAPE = 23.641
STAEformer_subway_out_50p__e50_h1_bis1:   All Steps RMSE = 37.929, MAE = 21.248, MASE = 0.658, MAPE = 21.574
STAEformer_subway_out_50p__e50_h1_bis2:   All Steps RMSE = 38.099, MAE = 21.351, MASE = 0.661, MAPE = 21.958
STAEformer_subway_out_50p__e50_h1_bis3:   All Steps RMSE = 37.732, MAE = 21.153, MASE = 0.655, MAPE = 22.056
STAEformer_subway_out_50p__e50_h1_bis4:   All Steps RMSE = 37.888, MAE = 21.224, MASE = 0.657, MAPE = 22.129
STAEformer_subway_out_50p__e50_h1_bis5:   All Steps RMSE = 37.852, MAE = 21.229, MASE = 0.658, MAPE = 22.510
STAEformer_subway_out_75p__e50_h1_bis1:   All Steps RMSE = 37.160, MAE = 20.702, MASE = 0.641, MAPE = 21.041
STAEformer_subway_out_75p__e50_h1_bis2:   All Steps RMSE = 37.323, MAE = 20.882, MASE = 0.647, MAPE = 21.567
STAEformer_subway_out_75p__e50_h1_bis3:   All Steps RMSE = 37.007, MAE = 20.558, MASE = 0.637, MAPE = 21.080
STAEformer_subway_out_75p__e50_h1_bis4:   All Steps RMSE = 37.161, MAE = 20.723, MASE = 0.642, MAPE = 21.740
STAEformer_subway_out_75p__e50_h1_bis5:   All Steps RMSE = 37.169, MAE = 20.857, MASE = 0.646, MAPE = 20.986
STAEformer_subway_out_100p__e50_h1_bis1:   All Steps RMSE = 37.327, MAE = 20.961, MASE = 0.649, MAPE = 22.792
STAEformer_subway_out_100p__e50_h1_bis2:   All Steps RMSE = 37.905, MAE = 21.370, MASE = 0.662, MAPE = 21.903
STAEformer_subway_out_100p__e50_h1_bis3:   All Steps RMSE = 37.301, MAE = 20.893, MASE = 0.647, MAPE = 20.864
STAEformer_subway_out_100p__e50_h1_bis4:   All Steps RMSE = 37.488, MAE = 20.992, MASE = 0.650, MAPE = 23.175
STAEformer_subway_out_100p__e50_h1_bis5:   All Steps RMSE = 37.254, MAE = 20.803, MASE = 0.644, MAPE = 21.474

STAEformer_subway_out_5p__e50_h4_bis1:   All Steps RMSE = 51.379, MAE = 29.066, MASE = 0.900, MAPE = 29.759
STAEformer_subway_out_5p__e50_h4_bis2:   All Steps RMSE = 52.541, MAE = 29.429, MASE = 0.911, MAPE = 29.960
STAEformer_subway_out_5p__e50_h4_bis3:   All Steps RMSE = 52.798, MAE = 30.104, MASE = 0.932, MAPE = 31.793
STAEformer_subway_out_5p__e50_h4_bis4:   All Steps RMSE = 52.631, MAE = 29.972, MASE = 0.928, MAPE = 33.092
STAEformer_subway_out_5p__e50_h4_bis5:   All Steps RMSE = 51.086, MAE = 29.004, MASE = 0.898, MAPE = 30.785
STAEformer_subway_out_10p__e50_h4_bis1:   All Steps RMSE = 50.566, MAE = 28.307, MASE = 0.877, MAPE = 29.360
STAEformer_subway_out_10p__e50_h4_bis2:   All Steps RMSE = 49.743, MAE = 27.915, MASE = 0.865, MAPE = 28.564
STAEformer_subway_out_10p__e50_h4_bis3:   All Steps RMSE = 51.137, MAE = 28.756, MASE = 0.891, MAPE = 33.526
STAEformer_subway_out_10p__e50_h4_bis4:   All Steps RMSE = 50.002, MAE = 28.038, MASE = 0.868, MAPE = 29.964
STAEformer_subway_out_10p__e50_h4_bis5:   All Steps RMSE = 49.514, MAE = 27.735, MASE = 0.859, MAPE = 28.889
STAEformer_subway_out_15p__e50_h4_bis1:   All Steps RMSE = 49.004, MAE = 27.144, MASE = 0.841, MAPE = 28.188
STAEformer_subway_out_15p__e50_h4_bis2:   All Steps RMSE = 49.293, MAE = 27.182, MASE = 0.842, MAPE = 28.059
STAEformer_subway_out_15p__e50_h4_bis3:   All Steps RMSE = 47.775, MAE = 26.394, MASE = 0.817, MAPE = 28.441
STAEformer_subway_out_15p__e50_h4_bis4:   All Steps RMSE = 48.882, MAE = 26.918, MASE = 0.834, MAPE = 27.388
STAEformer_subway_out_15p__e50_h4_bis5:   All Steps RMSE = 47.708, MAE = 26.568, MASE = 0.823, MAPE = 27.814
STAEformer_subway_out_25p__e50_h4_bis1:   All Steps RMSE = 45.912, MAE = 25.395, MASE = 0.786, MAPE = 26.449
STAEformer_subway_out_25p__e50_h4_bis2:   All Steps RMSE = 45.662, MAE = 25.511, MASE = 0.790, MAPE = 27.078
STAEformer_subway_out_25p__e50_h4_bis3:   All Steps RMSE = 47.321, MAE = 26.203, MASE = 0.812, MAPE = 26.201
STAEformer_subway_out_25p__e50_h4_bis4:   All Steps RMSE = 45.144, MAE = 25.158, MASE = 0.779, MAPE = 25.841
STAEformer_subway_out_25p__e50_h4_bis5:   All Steps RMSE = 45.296, MAE = 25.423, MASE = 0.787, MAPE = 28.325
STAEformer_subway_out_50p__e50_h4_bis1:   All Steps RMSE = 45.849, MAE = 25.063, MASE = 0.776, MAPE = 27.518
STAEformer_subway_out_50p__e50_h4_bis2:   All Steps RMSE = 44.984, MAE = 24.740, MASE = 0.766, MAPE = 25.668
STAEformer_subway_out_50p__e50_h4_bis3:   All Steps RMSE = 43.405, MAE = 24.067, MASE = 0.745, MAPE = 24.698
STAEformer_subway_out_50p__e50_h4_bis4:   All Steps RMSE = 45.098, MAE = 24.950, MASE = 0.773, MAPE = 26.670
STAEformer_subway_out_50p__e50_h4_bis5:   All Steps RMSE = 44.507, MAE = 24.813, MASE = 0.768, MAPE = 25.982
STAEformer_subway_out_75p__e50_h4_bis1:   All Steps RMSE = 43.221, MAE = 23.821, MASE = 0.738, MAPE = 25.148
STAEformer_subway_out_75p__e50_h4_bis2:   All Steps RMSE = 44.238, MAE = 24.381, MASE = 0.755, MAPE = 25.210
STAEformer_subway_out_75p__e50_h4_bis3:   All Steps RMSE = 43.045, MAE = 23.638, MASE = 0.732, MAPE = 25.748
STAEformer_subway_out_75p__e50_h4_bis4:   All Steps RMSE = 44.328, MAE = 24.543, MASE = 0.760, MAPE = 26.267
STAEformer_subway_out_75p__e50_h4_bis5:   All Steps RMSE = 43.254, MAE = 23.976, MASE = 0.743, MAPE = 24.850
STAEformer_subway_out_100p__e50_h4_bis1:   All Steps RMSE = 42.449, MAE = 23.336, MASE = 0.723, MAPE = 23.971
STAEformer_subway_out_100p__e50_h4_bis2:   All Steps RMSE = 42.888, MAE = 23.631, MASE = 0.732, MAPE = 23.721
STAEformer_subway_out_100p__e50_h4_bis3:   All Steps RMSE = 42.852, MAE = 23.405, MASE = 0.725, MAPE = 24.213
STAEformer_subway_out_100p__e50_h4_bis4:   All Steps RMSE = 43.347, MAE = 23.885, MASE = 0.740, MAPE = 23.967
STAEformer_subway_out_100p__e50_h4_bis5:   All Steps RMSE = 42.709, MAE = 23.208, MASE = 0.719, MAPE = 24.617

STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h1_bis1:   All Steps RMSE = 40.721, MAE = 23.630, MASE = 0.735, MAPE = 24.770
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h1_bis2:   All Steps RMSE = 40.104, MAE = 23.223, MASE = 0.723, MAPE = 23.828
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h1_bis3:   All Steps RMSE = 41.000, MAE = 23.509, MASE = 0.732, MAPE = 25.194
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h1_bis4:   All Steps RMSE = 39.894, MAE = 23.175, MASE = 0.721, MAPE = 24.917
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h1_bis5:   All Steps RMSE = 40.670, MAE = 23.600, MASE = 0.734, MAPE = 24.715
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h1_bis1:   All Steps RMSE = 39.470, MAE = 22.776, MASE = 0.709, MAPE = 23.676
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h1_bis2:   All Steps RMSE = 39.834, MAE = 23.124, MASE = 0.720, MAPE = 23.819
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h1_bis3:   All Steps RMSE = 39.701, MAE = 22.824, MASE = 0.710, MAPE = 23.600
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h1_bis4:   All Steps RMSE = 39.681, MAE = 23.013, MASE = 0.716, MAPE = 23.661
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h1_bis5:   All Steps RMSE = 39.327, MAE = 22.699, MASE = 0.706, MAPE = 23.787
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h1_bis1:   All Steps RMSE = 39.100, MAE = 22.404, MASE = 0.697, MAPE = 23.484
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h1_bis2:   All Steps RMSE = 38.795, MAE = 22.296, MASE = 0.694, MAPE = 23.738
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h1_bis3:   All Steps RMSE = 38.946, MAE = 22.283, MASE = 0.693, MAPE = 23.604
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h1_bis4:   All Steps RMSE = 38.645, MAE = 22.261, MASE = 0.693, MAPE = 24.027
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h1_bis5:   All Steps RMSE = 38.750, MAE = 22.229, MASE = 0.692, MAPE = 23.824
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h1_bis1:   All Steps RMSE = 37.858, MAE = 21.723, MASE = 0.676, MAPE = 23.765
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h1_bis2:   All Steps RMSE = 38.947, MAE = 22.354, MASE = 0.696, MAPE = 23.110
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h1_bis3:   All Steps RMSE = 38.284, MAE = 21.816, MASE = 0.679, MAPE = 22.979
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h1_bis4:   All Steps RMSE = 38.059, MAE = 21.700, MASE = 0.675, MAPE = 22.742
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h1_bis5:   All Steps RMSE = 38.476, MAE = 21.946, MASE = 0.683, MAPE = 22.314
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h1_bis1:   All Steps RMSE = 36.346, MAE = 20.511, MASE = 0.638, MAPE = 21.605
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h1_bis2:   All Steps RMSE = 36.552, MAE = 20.713, MASE = 0.645, MAPE = 23.037
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h1_bis3:   All Steps RMSE = 36.293, MAE = 20.509, MASE = 0.638, MAPE = 21.454
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h1_bis4:   All Steps RMSE = 36.812, MAE = 20.740, MASE = 0.645, MAPE = 22.027
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h1_bis5:   All Steps RMSE = 36.118, MAE = 20.345, MASE = 0.633, MAPE = 21.293
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h1_bis1:   All Steps RMSE = 37.201, MAE = 20.525, MASE = 0.639, MAPE = 21.317
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h1_bis2:   All Steps RMSE = 35.984, MAE = 20.113, MASE = 0.626, MAPE = 20.609
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h1_bis3:   All Steps RMSE = 36.809, MAE = 20.359, MASE = 0.633, MAPE = 20.666
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h1_bis4:   All Steps RMSE = 35.793, MAE = 20.043, MASE = 0.624, MAPE = 20.814
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h1_bis5:   All Steps RMSE = 35.759, MAE = 19.962, MASE = 0.621, MAPE = 20.871
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h1_bis1:   All Steps RMSE = 36.127, MAE = 19.922, MASE = 0.620, MAPE = 20.570
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h1_bis2:   All Steps RMSE = 35.784, MAE = 19.874, MASE = 0.618, MAPE = 20.293
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h1_bis3:   All Steps RMSE = 35.444, MAE = 19.730, MASE = 0.614, MAPE = 20.457
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h1_bis4:   All Steps RMSE = 35.643, MAE = 19.807, MASE = 0.616, MAPE = 20.539
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h1_bis5:   All Steps RMSE = 35.432, MAE = 19.791, MASE = 0.616, MAPE = 20.616

STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h4_bis1:   All Steps RMSE = 50.185, MAE = 28.870, MASE = 0.899, MAPE = 30.743
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h4_bis2:   All Steps RMSE = 50.690, MAE = 28.720, MASE = 0.894, MAPE = 28.664
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h4_bis3:   All Steps RMSE = 48.714, MAE = 27.920, MASE = 0.869, MAPE = 30.013
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h4_bis4:   All Steps RMSE = 49.076, MAE = 27.840, MASE = 0.867, MAPE = 28.298
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_5p__e50_h4_bis5:   All Steps RMSE = 48.629, MAE = 28.101, MASE = 0.875, MAPE = 32.660
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h4_bis1:   All Steps RMSE = 48.565, MAE = 27.419, MASE = 0.854, MAPE = 29.281
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h4_bis2:   All Steps RMSE = 47.374, MAE = 26.895, MASE = 0.837, MAPE = 27.677
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h4_bis3:   All Steps RMSE = 48.051, MAE = 27.250, MASE = 0.848, MAPE = 28.856
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h4_bis4:   All Steps RMSE = 46.984, MAE = 26.705, MASE = 0.831, MAPE = 28.552
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_10p__e50_h4_bis5:   All Steps RMSE = 47.794, MAE = 26.911, MASE = 0.838, MAPE = 28.084
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h4_bis1:   All Steps RMSE = 46.296, MAE = 26.216, MASE = 0.816, MAPE = 26.430
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h4_bis2:   All Steps RMSE = 46.011, MAE = 25.945, MASE = 0.808, MAPE = 27.590
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h4_bis3:   All Steps RMSE = 47.701, MAE = 26.840, MASE = 0.836, MAPE = 29.270
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h4_bis4:   All Steps RMSE = 46.803, MAE = 26.295, MASE = 0.819, MAPE = 27.407
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_15p__e50_h4_bis5:   All Steps RMSE = 45.888, MAE = 25.817, MASE = 0.804, MAPE = 26.439
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h4_bis1:   All Steps RMSE = 44.454, MAE = 24.942, MASE = 0.776, MAPE = 25.592
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h4_bis2:   All Steps RMSE = 43.510, MAE = 24.559, MASE = 0.765, MAPE = 27.006
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h4_bis3:   All Steps RMSE = 43.848, MAE = 24.694, MASE = 0.769, MAPE = 26.178
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h4_bis4:   All Steps RMSE = 44.316, MAE = 24.703, MASE = 0.769, MAPE = 26.473
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_25p__e50_h4_bis5:   All Steps RMSE = 43.734, MAE = 24.545, MASE = 0.764, MAPE = 25.935
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h4_bis1:   All Steps RMSE = 41.984, MAE = 23.571, MASE = 0.734, MAPE = 24.615
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h4_bis2:   All Steps RMSE = 42.655, MAE = 23.578, MASE = 0.734, MAPE = 24.933
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h4_bis3:   All Steps RMSE = 41.491, MAE = 23.056, MASE = 0.718, MAPE = 23.767
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h4_bis4:   All Steps RMSE = 41.747, MAE = 23.232, MASE = 0.723, MAPE = 24.448
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_50p__e50_h4_bis5:   All Steps RMSE = 42.019, MAE = 23.351, MASE = 0.727, MAPE = 24.588
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h4_bis1:   All Steps RMSE = 40.827, MAE = 22.675, MASE = 0.706, MAPE = 23.407
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h4_bis2:   All Steps RMSE = 40.678, MAE = 22.677, MASE = 0.706, MAPE = 24.336
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h4_bis3:   All Steps RMSE = 40.752, MAE = 22.562, MASE = 0.702, MAPE = 23.176
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h4_bis4:   All Steps RMSE = 40.861, MAE = 22.600, MASE = 0.704, MAPE = 23.441
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_75p__e50_h4_bis5:   All Steps RMSE = 41.900, MAE = 23.134, MASE = 0.720, MAPE = 23.495
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h4_bis1:   All Steps RMSE = 40.467, MAE = 22.302, MASE = 0.694, MAPE = 22.889
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h4_bis2:   All Steps RMSE = 40.688, MAE = 22.548, MASE = 0.702, MAPE = 22.586
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h4_bis3:   All Steps RMSE = 41.312, MAE = 22.944, MASE = 0.714, MAPE = 23.009
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h4_bis4:   All Steps RMSE = 40.386, MAE = 22.517, MASE = 0.701, MAPE = 22.682
STAEformer_subway_out_subway_in_calendar_early_fusion_shared_embedding_100p__e50_h4_bis5:   All Steps RMSE = 40.507, MAE = 22.370, MASE = 0.696, MAPE = 23.222

STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h1_bis1:   All Steps RMSE = 40.006, MAE = 23.290, MASE = 0.725, MAPE = 25.344
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h1_bis2:   All Steps RMSE = 39.786, MAE = 23.057, MASE = 0.717, MAPE = 23.872
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h1_bis3:   All Steps RMSE = 40.150, MAE = 23.356, MASE = 0.727, MAPE = 24.473
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h1_bis4:   All Steps RMSE = 39.994, MAE = 23.179, MASE = 0.721, MAPE = 25.639
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h1_bis5:   All Steps RMSE = 40.572, MAE = 23.369, MASE = 0.727, MAPE = 25.242
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h1_bis1:   All Steps RMSE = 39.280, MAE = 22.781, MASE = 0.709, MAPE = 23.760
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h1_bis2:   All Steps RMSE = 39.933, MAE = 22.987, MASE = 0.715, MAPE = 24.117
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h1_bis3:   All Steps RMSE = 39.695, MAE = 22.964, MASE = 0.715, MAPE = 23.370
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h1_bis4:   All Steps RMSE = 39.778, MAE = 22.965, MASE = 0.715, MAPE = 23.617
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h1_bis5:   All Steps RMSE = 39.694, MAE = 22.902, MASE = 0.713, MAPE = 23.844
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h1_bis1:   All Steps RMSE = 38.721, MAE = 22.328, MASE = 0.695, MAPE = 24.310
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h1_bis2:   All Steps RMSE = 38.502, MAE = 22.259, MASE = 0.693, MAPE = 23.562
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h1_bis3:   All Steps RMSE = 38.901, MAE = 22.253, MASE = 0.692, MAPE = 22.707
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h1_bis4:   All Steps RMSE = 39.253, MAE = 22.390, MASE = 0.697, MAPE = 23.441
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h1_bis5:   All Steps RMSE = 38.892, MAE = 22.212, MASE = 0.691, MAPE = 23.732
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h1_bis1:   All Steps RMSE = 37.158, MAE = 21.254, MASE = 0.661, MAPE = 22.991
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h1_bis2:   All Steps RMSE = 37.840, MAE = 21.690, MASE = 0.675, MAPE = 22.925
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h1_bis3:   All Steps RMSE = 38.463, MAE = 22.035, MASE = 0.686, MAPE = 23.614
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h1_bis4:   All Steps RMSE = 37.763, MAE = 21.576, MASE = 0.671, MAPE = 22.597
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h1_bis5:   All Steps RMSE = 37.989, MAE = 21.679, MASE = 0.675, MAPE = 22.349
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h1_bis1:   All Steps RMSE = 36.322, MAE = 20.423, MASE = 0.635, MAPE = 21.278
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h1_bis2:   All Steps RMSE = 36.153, MAE = 20.351, MASE = 0.633, MAPE = 21.882
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h1_bis3:   All Steps RMSE = 36.039, MAE = 20.305, MASE = 0.632, MAPE = 21.474
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h1_bis4:   All Steps RMSE = 36.361, MAE = 20.596, MASE = 0.641, MAPE = 21.251
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h1_bis5:   All Steps RMSE = 37.111, MAE = 20.751, MASE = 0.646, MAPE = 21.291
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h1_bis1:   All Steps RMSE = 36.444, MAE = 20.265, MASE = 0.631, MAPE = 20.898
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h1_bis2:   All Steps RMSE = 35.772, MAE = 19.964, MASE = 0.621, MAPE = 20.632
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h1_bis3:   All Steps RMSE = 35.949, MAE = 20.030, MASE = 0.623, MAPE = 20.792
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h1_bis4:   All Steps RMSE = 36.159, MAE = 20.083, MASE = 0.625, MAPE = 20.930
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h1_bis5:   All Steps RMSE = 35.949, MAE = 20.092, MASE = 0.625, MAPE = 20.906
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h1_bis1:   All Steps RMSE = 35.640, MAE = 19.751, MASE = 0.615, MAPE = 20.762
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h1_bis2:   All Steps RMSE = 35.667, MAE = 19.784, MASE = 0.616, MAPE = 20.413
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h1_bis3:   All Steps RMSE = 35.925, MAE = 19.901, MASE = 0.619, MAPE = 20.657
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h1_bis4:   All Steps RMSE = 35.940, MAE = 19.967, MASE = 0.621, MAPE = 20.739
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h1_bis5:   All Steps RMSE = 35.877, MAE = 19.803, MASE = 0.616, MAPE = 20.369

STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h4_bis1:   All Steps RMSE = 49.839, MAE = 28.463, MASE = 0.886, MAPE = 29.732
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h4_bis2:   All Steps RMSE = 49.310, MAE = 28.059, MASE = 0.874, MAPE = 29.989
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h4_bis3:   All Steps RMSE = 49.721, MAE = 28.408, MASE = 0.884, MAPE = 30.329
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h4_bis4:   All Steps RMSE = 49.095, MAE = 28.200, MASE = 0.878, MAPE = 30.347
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_5p__e50_h4_bis5:   All Steps RMSE = 49.501, MAE = 28.421, MASE = 0.885, MAPE = 29.907
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h4_bis1:   All Steps RMSE = 47.513, MAE = 27.134, MASE = 0.845, MAPE = 27.556
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h4_bis2:   All Steps RMSE = 48.717, MAE = 27.391, MASE = 0.853, MAPE = 29.572
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h4_bis3:   All Steps RMSE = 47.987, MAE = 27.439, MASE = 0.854, MAPE = 32.128
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h4_bis4:   All Steps RMSE = 48.582, MAE = 27.465, MASE = 0.855, MAPE = 28.370
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_10p__e50_h4_bis5:   All Steps RMSE = 48.134, MAE = 27.152, MASE = 0.845, MAPE = 28.049
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h4_bis1:   All Steps RMSE = 45.402, MAE = 25.800, MASE = 0.803, MAPE = 28.927
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h4_bis2:   All Steps RMSE = 45.901, MAE = 25.829, MASE = 0.804, MAPE = 27.264
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h4_bis3:   All Steps RMSE = 47.267, MAE = 26.779, MASE = 0.834, MAPE = 28.023
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h4_bis4:   All Steps RMSE = 45.888, MAE = 26.149, MASE = 0.814, MAPE = 27.632
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_15p__e50_h4_bis5:   All Steps RMSE = 44.905, MAE = 25.529, MASE = 0.795, MAPE = 27.584
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h4_bis1:   All Steps RMSE = 43.676, MAE = 24.512, MASE = 0.763, MAPE = 25.681
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h4_bis2:   All Steps RMSE = 44.771, MAE = 25.394, MASE = 0.791, MAPE = 26.731
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h4_bis3:   All Steps RMSE = 44.286, MAE = 24.892, MASE = 0.775, MAPE = 25.599
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h4_bis4:   All Steps RMSE = 44.176, MAE = 24.732, MASE = 0.770, MAPE = 26.252
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_25p__e50_h4_bis5:   All Steps RMSE = 44.532, MAE = 24.975, MASE = 0.777, MAPE = 26.816
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h4_bis1:   All Steps RMSE = 42.817, MAE = 23.773, MASE = 0.740, MAPE = 24.358
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h4_bis2:   All Steps RMSE = 41.511, MAE = 23.322, MASE = 0.726, MAPE = 24.371
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h4_bis3:   All Steps RMSE = 41.320, MAE = 23.171, MASE = 0.721, MAPE = 24.407
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h4_bis4:   All Steps RMSE = 41.865, MAE = 23.457, MASE = 0.730, MAPE = 24.001
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_50p__e50_h4_bis5:   All Steps RMSE = 43.170, MAE = 24.004, MASE = 0.747, MAPE = 24.756
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h4_bis1:   All Steps RMSE = 41.997, MAE = 23.202, MASE = 0.722, MAPE = 23.447
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h4_bis2:   All Steps RMSE = 41.843, MAE = 23.386, MASE = 0.728, MAPE = 23.409
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h4_bis3:   All Steps RMSE = 41.429, MAE = 23.037, MASE = 0.717, MAPE = 23.760
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h4_bis4:   All Steps RMSE = 41.408, MAE = 23.080, MASE = 0.719, MAPE = 23.355
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_75p__e50_h4_bis5:   All Steps RMSE = 40.905, MAE = 22.671, MASE = 0.706, MAPE = 23.445
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h4_bis1:   All Steps RMSE = 40.983, MAE = 22.599, MASE = 0.704, MAPE = 23.115
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h4_bis2:   All Steps RMSE = 40.494, MAE = 22.291, MASE = 0.694, MAPE = 23.517
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h4_bis3:   All Steps RMSE = 40.683, MAE = 22.715, MASE = 0.707, MAPE = 23.261
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h4_bis4:   All Steps RMSE = 40.690, MAE = 22.480, MASE = 0.700, MAPE = 23.390
STAEformer_subway_out_subway_in_weather_calendar_early_fusion_shared_embedding_repeat_t_proj_100p__e50_h4_bis5:   All Steps RMSE = 41.074, MAE = 22.711, MASE = 0.707, MAPE = 23.014
"""


# STAEformer_bike_out_5p__e50_h1_bis1:   All Steps RMSE = 4.720, MAE = 3.027, MASE = 0.786, MAPE = 50.174
# STAEformer_bike_out_5p__e50_h1_bis2:   All Steps RMSE = 4.784, MAE = 3.099, MASE = 0.805, MAPE = 54.013
# STAEformer_bike_out_5p__e50_h1_bis3:   All Steps RMSE = 4.701, MAE = 3.015, MASE = 0.783, MAPE = 49.846
# STAEformer_bike_out_5p__e50_h1_bis4:   All Steps RMSE = 4.682, MAE = 3.035, MASE = 0.788, MAPE = 52.196
# STAEformer_bike_out_5p__e50_h1_bis5:   All Steps RMSE = 4.776, MAE = 3.041, MASE = 0.790, MAPE = 49.694
# STAEformer_bike_out_10p__e50_h1_bis1:   All Steps RMSE = 4.694, MAE = 3.017, MASE = 0.783, MAPE = 51.144
# STAEformer_bike_out_10p__e50_h1_bis2:   All Steps RMSE = 4.690, MAE = 3.002, MASE = 0.779, MAPE = 50.280
# STAEformer_bike_out_10p__e50_h1_bis3:   All Steps RMSE = 4.640, MAE = 2.991, MASE = 0.777, MAPE = 51.115
# STAEformer_bike_out_10p__e50_h1_bis4:   All Steps RMSE = 4.690, MAE = 3.026, MASE = 0.786, MAPE = 52.008
# STAEformer_bike_out_10p__e50_h1_bis5:   All Steps RMSE = 4.694, MAE = 2.981, MASE = 0.774, MAPE = 48.541
# STAEformer_bike_out_15p__e50_h1_bis1:   All Steps RMSE = 4.579, MAE = 2.937, MASE = 0.762, MAPE = 48.496
# STAEformer_bike_out_15p__e50_h1_bis2:   All Steps RMSE = 4.596, MAE = 2.952, MASE = 0.766, MAPE = 49.102
# STAEformer_bike_out_15p__e50_h1_bis3:   All Steps RMSE = 4.554, MAE = 2.927, MASE = 0.760, MAPE = 48.297
# STAEformer_bike_out_15p__e50_h1_bis4:   All Steps RMSE = 4.547, MAE = 2.920, MASE = 0.758, MAPE = 48.248
# STAEformer_bike_out_15p__e50_h1_bis5:   All Steps RMSE = 4.557, MAE = 2.946, MASE = 0.765, MAPE = 49.956
# STAEformer_bike_out_25p__e50_h1_bis1:   All Steps RMSE = 4.465, MAE = 2.908, MASE = 0.755, MAPE = 50.202
# STAEformer_bike_out_25p__e50_h1_bis2:   All Steps RMSE = 4.466, MAE = 2.877, MASE = 0.747, MAPE = 46.415
# STAEformer_bike_out_25p__e50_h1_bis3:   All Steps RMSE = 4.464, MAE = 2.877, MASE = 0.747, MAPE = 47.623
# STAEformer_bike_out_25p__e50_h1_bis4:   All Steps RMSE = 4.508, MAE = 2.887, MASE = 0.750, MAPE = 45.302
# STAEformer_bike_out_25p__e50_h1_bis5:   All Steps RMSE = 4.497, MAE = 2.892, MASE = 0.751, MAPE = 47.577
# STAEformer_bike_out_50p__e50_h1_bis1:   All Steps RMSE = 4.407, MAE = 2.825, MASE = 0.733, MAPE = 43.810
# STAEformer_bike_out_50p__e50_h1_bis2:   All Steps RMSE = 4.384, MAE = 2.831, MASE = 0.735, MAPE = 45.554
# STAEformer_bike_out_50p__e50_h1_bis3:   All Steps RMSE = 4.421, MAE = 2.843, MASE = 0.738, MAPE = 45.456
# STAEformer_bike_out_50p__e50_h1_bis4:   All Steps RMSE = 4.416, MAE = 2.843, MASE = 0.738, MAPE = 45.566
# STAEformer_bike_out_50p__e50_h1_bis5:   All Steps RMSE = 4.387, MAE = 2.825, MASE = 0.734, MAPE = 45.338
# STAEformer_bike_out_75p__e50_h1_bis1:   All Steps RMSE = 4.306, MAE = 2.790, MASE = 0.725, MAPE = 45.987
# STAEformer_bike_out_75p__e50_h1_bis2:   All Steps RMSE = 4.310, MAE = 2.788, MASE = 0.724, MAPE = 46.256
# STAEformer_bike_out_75p__e50_h1_bis3:   All Steps RMSE = 4.327, MAE = 2.800, MASE = 0.727, MAPE = 46.167
# STAEformer_bike_out_75p__e50_h1_bis4:   All Steps RMSE = 4.401, MAE = 2.842, MASE = 0.738, MAPE = 46.013
# STAEformer_bike_out_75p__e50_h1_bis5:   All Steps RMSE = 4.330, MAE = 2.795, MASE = 0.726, MAPE = 45.427
# STAEformer_bike_out_100p__e50_h1_bis1:   All Steps RMSE = 4.279, MAE = 2.792, MASE = 0.725, MAPE = 47.208
# STAEformer_bike_out_100p__e50_h1_bis2:   All Steps RMSE = 4.304, MAE = 2.772, MASE = 0.720, MAPE = 45.113
# STAEformer_bike_out_100p__e50_h1_bis3:   All Steps RMSE = 4.312, MAE = 2.782, MASE = 0.722, MAPE = 45.359
# STAEformer_bike_out_100p__e50_h1_bis4:   All Steps RMSE = 4.293, MAE = 2.773, MASE = 0.720, MAPE = 45.124
# STAEformer_bike_out_100p__e50_h1_bis5:   All Steps RMSE = 4.298, MAE = 2.770, MASE = 0.719, MAPE = 45.138

# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis1:   All Steps RMSE = 4.791, MAE = 3.073, MASE = 0.798, MAPE = 51.685
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis2:   All Steps RMSE = 4.794, MAE = 3.056, MASE = 0.794, MAPE = 50.121
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis3:   All Steps RMSE = 4.706, MAE = 3.005, MASE = 0.780, MAPE = 48.619
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis4:   All Steps RMSE = 4.724, MAE = 3.013, MASE = 0.782, MAPE = 49.553
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h1_bis5:   All Steps RMSE = 4.772, MAE = 3.056, MASE = 0.793, MAPE = 51.590
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis1:   All Steps RMSE = 4.620, MAE = 2.961, MASE = 0.769, MAPE = 47.793
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis2:   All Steps RMSE = 4.633, MAE = 2.986, MASE = 0.775, MAPE = 50.352
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis3:   All Steps RMSE = 4.698, MAE = 2.991, MASE = 0.777, MAPE = 48.808
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis4:   All Steps RMSE = 4.592, MAE = 2.937, MASE = 0.763, MAPE = 47.602
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h1_bis5:   All Steps RMSE = 4.571, MAE = 2.929, MASE = 0.760, MAPE = 47.095
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis1:   All Steps RMSE = 4.562, MAE = 2.928, MASE = 0.760, MAPE = 46.987
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis2:   All Steps RMSE = 4.573, MAE = 2.985, MASE = 0.775, MAPE = 52.338
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis3:   All Steps RMSE = 4.598, MAE = 2.957, MASE = 0.768, MAPE = 48.848
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis4:   All Steps RMSE = 4.560, MAE = 2.940, MASE = 0.763, MAPE = 49.354
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h1_bis5:   All Steps RMSE = 4.590, MAE = 2.937, MASE = 0.763, MAPE = 48.376
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis1:   All Steps RMSE = 4.479, MAE = 2.894, MASE = 0.751, MAPE = 47.739
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis2:   All Steps RMSE = 4.542, MAE = 2.920, MASE = 0.758, MAPE = 46.856
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis3:   All Steps RMSE = 4.492, MAE = 2.921, MASE = 0.758, MAPE = 49.865
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis4:   All Steps RMSE = 4.490, MAE = 2.894, MASE = 0.751, MAPE = 47.751
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h1_bis5:   All Steps RMSE = 4.487, MAE = 2.917, MASE = 0.757, MAPE = 49.215
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis1:   All Steps RMSE = 4.340, MAE = 2.811, MASE = 0.730, MAPE = 46.289
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis2:   All Steps RMSE = 4.347, MAE = 2.803, MASE = 0.728, MAPE = 44.968
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis3:   All Steps RMSE = 4.374, MAE = 2.827, MASE = 0.734, MAPE = 45.965
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis4:   All Steps RMSE = 4.386, MAE = 2.829, MASE = 0.735, MAPE = 45.230
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h1_bis5:   All Steps RMSE = 4.365, MAE = 2.831, MASE = 0.735, MAPE = 46.337
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis1:   All Steps RMSE = 4.374, MAE = 2.822, MASE = 0.733, MAPE = 46.006
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis2:   All Steps RMSE = 4.429, MAE = 2.828, MASE = 0.734, MAPE = 43.893
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis3:   All Steps RMSE = 4.327, MAE = 2.798, MASE = 0.726, MAPE = 44.958
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis4:   All Steps RMSE = 4.467, MAE = 2.895, MASE = 0.752, MAPE = 48.160
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h1_bis5:   All Steps RMSE = 4.347, MAE = 2.805, MASE = 0.728, MAPE = 44.538

# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis1:   All Steps RMSE = 5.576, MAE = 3.508, MASE = 0.911, MAPE = 60.824
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis2:   All Steps RMSE = 5.663, MAE = 3.606, MASE = 0.936, MAPE = 64.600
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis3:   All Steps RMSE = 5.692, MAE = 3.551, MASE = 0.922, MAPE = 57.731
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis4:   All Steps RMSE = 5.594, MAE = 3.545, MASE = 0.921, MAPE = 62.515
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_5p__e50_h4_bis5:   All Steps RMSE = 5.672, MAE = 3.590, MASE = 0.932, MAPE = 64.082
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis1:   All Steps RMSE = 5.554, MAE = 3.470, MASE = 0.901, MAPE = 59.614
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis2:   All Steps RMSE = 5.383, MAE = 3.380, MASE = 0.878, MAPE = 57.317
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis3:   All Steps RMSE = 5.409, MAE = 3.410, MASE = 0.885, MAPE = 59.365
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis4:   All Steps RMSE = 5.365, MAE = 3.365, MASE = 0.874, MAPE = 57.421
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_10p__e50_h4_bis5:   All Steps RMSE = 5.480, MAE = 3.433, MASE = 0.891, MAPE = 58.580
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis1:   All Steps RMSE = 5.348, MAE = 3.371, MASE = 0.875, MAPE = 59.243
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis2:   All Steps RMSE = 5.242, MAE = 3.285, MASE = 0.853, MAPE = 54.431
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis3:   All Steps RMSE = 5.329, MAE = 3.326, MASE = 0.864, MAPE = 55.742
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis4:   All Steps RMSE = 5.182, MAE = 3.222, MASE = 0.837, MAPE = 50.272
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_15p__e50_h4_bis5:   All Steps RMSE = 5.266, MAE = 3.282, MASE = 0.852, MAPE = 54.201
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis1:   All Steps RMSE = 5.167, MAE = 3.214, MASE = 0.835, MAPE = 50.877
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis2:   All Steps RMSE = 5.081, MAE = 3.173, MASE = 0.824, MAPE = 49.330
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis3:   All Steps RMSE = 5.064, MAE = 3.184, MASE = 0.827, MAPE = 52.403
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis4:   All Steps RMSE = 5.135, MAE = 3.207, MASE = 0.833, MAPE = 51.021
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_25p__e50_h4_bis5:   All Steps RMSE = 5.106, MAE = 3.197, MASE = 0.830, MAPE = 51.883
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis1:   All Steps RMSE = 4.989, MAE = 3.129, MASE = 0.812, MAPE = 50.334
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis2:   All Steps RMSE = 4.992, MAE = 3.131, MASE = 0.813, MAPE = 50.656
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis3:   All Steps RMSE = 4.943, MAE = 3.098, MASE = 0.804, MAPE = 48.687
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis4:   All Steps RMSE = 4.980, MAE = 3.121, MASE = 0.810, MAPE = 50.005
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_50p__e50_h4_bis5:   All Steps RMSE = 4.973, MAE = 3.107, MASE = 0.807, MAPE = 49.755
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis1:   All Steps RMSE = 5.018, MAE = 3.186, MASE = 0.827, MAPE = 54.297
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis2:   All Steps RMSE = 5.019, MAE = 3.127, MASE = 0.812, MAPE = 50.019
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis3:   All Steps RMSE = 4.931, MAE = 3.091, MASE = 0.803, MAPE = 49.343
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis4:   All Steps RMSE = 4.969, MAE = 3.129, MASE = 0.813, MAPE = 51.952
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_75p__e50_h4_bis5:   All Steps RMSE = 4.929, MAE = 3.107, MASE = 0.807, MAPE = 52.192
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis1:   All Steps RMSE = 4.805, MAE = 3.065, MASE = 0.796, MAPE = 52.220
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis2:   All Steps RMSE = 4.861, MAE = 3.055, MASE = 0.793, MAPE = 49.522
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis3:   All Steps RMSE = 4.874, MAE = 3.069, MASE = 0.797, MAPE = 51.375
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis4:   All Steps RMSE = 4.868, MAE = 3.048, MASE = 0.791, MAPE = 49.263
# STAEformer_bike_out_weather_calendar_early_fusion_repeat_t_proj_100p__e50_h4_bis5:   All Steps RMSE = 4.829, MAE = 3.038, MASE = 0.789, MAPE = 49.196