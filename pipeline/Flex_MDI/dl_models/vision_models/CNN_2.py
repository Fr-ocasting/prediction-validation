
# Input Layer (H,W,C)
# Conv2D(3,3) (50,50,64)
# BN          (50,50,64)                                     Weather
# Conv2D(3,3) (25,25,128)                                    Calendar 
# Global AvgPool (128)                                        ...
#           |---------Concat avec d'autres inputs -------------|
# Multiple option (64)???
# Ouptut (1)