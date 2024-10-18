import torch 
import torch.nn as nn
import torch.nn.functional as F

def assert_dim_tensor(input_tensor,spatial_window_size):
    assert spatial_window_size % 2 == 1, f"Spatial_window_size {spatial_window_size} Must be odd for a correct centring"
    if input_tensor.dim() == 3:
        input_tensor= input_tensor.unsqueeze(0)
    elif input_tensor.dim() > 3:
        input_tensor = input_tensor.unsqueeze(-4)
        
    else:
        raise ValueError(f'Input tensor dimension {input_tensor.dim()} is not sufficient (require at least 3 dim [H,W,T])')
    return input_tensor

class ExponentialMovingAverage(object):
    def __init__(self,input_tensor, temporal_alpha, spatial_alpha, temporal_window_size = 3,spatial_window_size = 3):
        self.input_tensor_dim = assert_dim_tensor(input_tensor,spatial_window_size).dim()
        self.temporal_alpha = temporal_alpha
        self.spatial_alpha = spatial_alpha
        self.temporal_window_size = temporal_window_size
        self.spatial_window_size = spatial_window_size
        self.load_EMA()
        
        
    def load_EMA(self):
        # Define EMA weights for past and current frames only (causal filter)
        others_kernel_size = [1]*(self.input_tensor_dim -3)

        kernel_size = (self.spatial_window_size, self.spatial_window_size, self.temporal_window_size)
        
        # Get filter Weights
        temporal_weights = torch.tensor([(1 - self.temporal_alpha) ** (i) for i in range(self.temporal_window_size)], dtype=torch.float32)
        spatial_weights_i = [(1 - self.spatial_alpha) ** (i+1) for i in range(self.spatial_window_size//2)]
        spatial_weights = torch.tensor(spatial_weights_i[::-1]+[1]+spatial_weights_i, dtype=torch.float32) # centered weights
        spatial_weights = torch.matmul(spatial_weights.reshape(-1,1),spatial_weights.reshape(1,-1))
        kernel = torch.einsum('np,k->npk', spatial_weights,temporal_weights)
        
        # Norm filter 
        kernel = kernel/kernel.sum()
        # Apply 3D convolution with padding to keep causal structure (only past frames)
        conv = nn.Conv3d(1,1,
                         kernel_size = kernel_size,
                         padding=(self.spatial_window_size//2, self.spatial_window_size//2, 0), # symetrical pad for spatial conv
                         bias = False)  
        conv.weight = nn.Parameter(kernel.reshape(*others_kernel_size,*kernel_size))
        self.conv = conv
        
    def compute_ema(self,input_tensor):
        """
        Applies EMA on the spatial (2nd to last dimensions) and temporal (last dimension)
        of a 5D tensor using a 3D convolution. It considers only past and current values
        
        Args :
        ----------
        input_tensor : At least 3D PyTorch Tensor (*,H, W, T), for which we are adding a channel dim C = 1
            C:  channel dim (expected 1), H,W image size, T temporal dim 
        alpha : Smoothing factor for EMA (0 < alpha <= 1).

        Returns:
        ----------
            5D PyTorch tensor with EMA applied along the last 3 dimensions (spatial + temporal).
        """
        input_tensor = assert_dim_tensor(input_tensor,self.spatial_window_size)

        # Asymetrical padding 
        input_padded = F.pad(input_tensor.float(), pad=(self.temporal_window_size - 1, 0, 0, 0, 0, 0))  # (pad_time_start, pad_time_end, pad_width_start, pad_width_end, pad_height_start, pad_height_end)
        smoothed_imput_tensor = self.conv(input_padded)
        smoothed_imput_tensor = smoothed_imput_tensor.squeeze()
        return(smoothed_imput_tensor)
    


if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    def plot3D(tensor):
        # Créer les coordonnées pour la visualisation
        x = torch.arange(tensor.size(0))
        y = torch.arange(tensor.size(1))
        z = torch.arange(tensor.size(2))

        # Créer une grille pour X, Y, Z
        X, Y, Z = torch.meshgrid(x, y, z, indexing="ij")

        # Convertir les données en numpy pour matplotlib
        X_np = X.numpy()
        Y_np = Y.numpy()
        Z_np = Z.numpy()
        tensor_np = tensor.detach().numpy()

        # Visualisation avec matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Tracer les points 3D avec un colormap viridis
        img = ax.scatter(X_np, Y_np, Z_np, c=tensor_np.flatten(), cmap='viridis')

        # Ajouter une barre de couleur
        fig.colorbar(img)

        plt.show()

    alpha = 0.8
    tensor_test = torch.tensor([1]*30*30*10).reshape(1,30,30,10)
    tensor_test[...,-5:] = torch.Tensor([10]*30*30*5).reshape(1,30,30,5)
    EMA = ExponentialMovingAverage(tensor_test, 
                             temporal_alpha=alpha, 
                             spatial_alpha=alpha, 
                             temporal_window_size = 3,
                             spatial_window_size = 5
                            )
    smoothed_tensor_test = EMA.compute_ema(tensor_test)


    plot3D(EMA.conv.weight.squeeze())
    plot3D(tensor_test.squeeze())
    plot3D(smoothed_tensor_test)
    plot3D(smoothed_tensor_test[2:-2,2:-2,:])