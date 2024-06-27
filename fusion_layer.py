#our method
import torch
import torch.nn as nn
import torch.nn.functional as F
class FusionLayer(torch.nn.Module):
    def __init__(self, x1_dim, x2_dim):
        super(FusionLayer, self).__init__()
        self.W = nn.Linear(x1_dim, x2_dim)
    
    def forward(self,x1, x2):
        # Project x2 to x1's space
        x2= self.W(x2)
        # Element wise mul
        z = x1*x2
        z = F.tanh(z)
        return x1*z