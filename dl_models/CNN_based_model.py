from torch import nn

class cnn_perso(nn.Module):
    def __init__(self,c_in, h_dim, c_out, kernel_size, L, padding = 0):
        super().__init__()

        dim_last_conv = 2*h_dim

        self.c_in = c_in
        self.c_out = c_out
        self.kernel_size = kernel_size
        self.padding = padding
        self.name = f'CNN_h{h_dim}_L{L}_2layer'
        h_out = (dim_last_conv*(L-2))

        self.conv1 = nn.Conv1d(c_in, h_dim, kernel_size,padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(h_dim, 2*h_dim, kernel_size, padding=padding)
        self.flatten = nn.Flatten()
        
        self.dense_out1 = nn.Linear(h_out, 32)
        self.dense_out2 = nn.Linear(32, c_out)
    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.flatten(x)
        x = self.dense_out1(x)
        x = self.dense_out2(x)
        return(x)