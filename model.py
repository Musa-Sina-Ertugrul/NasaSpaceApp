import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from kan import KAN,KANLinear

class DataSet(Dataset):
    
    def __init__(self,X,Y,device:str='cuda') -> None:
        super().__init__()
        self.X = torch.tensor(X,dtype=torch.float32,device=device)
        self.Y = torch.tensor(Y,dtype=torch.float32,device=device)
        self.X[self.X==0] = 1e-3
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.Y[index]

class MFP(torch.nn.Module):
        
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.leaky_relu = torch.nn.LeakyReLU()
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.mish = torch.nn.Mish()
        self.silu = torch.nn.SiLU()
        self.selu = torch.nn.SELU()
        self.elu = torch.nn.ELU()
        self.gelu = torch.nn.GELU()
        self.softplus = torch.nn.Softplus()
        self.softsign = torch.nn.Softsign()
        self.softshrink = torch.nn.Softshrink()
        
        self.leaky_relu_linear = torch.nn.Linear(285, 285)
        self.relu_linear = torch.nn.Linear(285, 285)
        self.tanh_linear = torch.nn.Linear(285,285)
        self.sigmoid_linear = torch.nn.Linear(285, 285)
        self.mish_linear = torch.nn.Linear(285, 285)
        self.selu_linear = torch.nn.Linear(285,285)
        self.silu_linear = torch.nn.Linear(285,285)
        self.elu_linear = torch.nn.Linear(285, 285)
        self.gelu_linear = torch.nn.Linear(285,285)
        self.softplus_linear = torch.nn.Linear(285, 285)
        self.softsign_linear = torch.nn.Linear(285, 285)
        self.softshrink_linear = torch.nn.Linear(285, 285)
        
        self.layer_norm = torch.nn.LayerNorm(12*285)
        
        self.kan = KAN([12*285,8,64,8],grid_size=10,spline_order=5)
    
    def forward(self,x):
        
        x_relu = self.relu_linear(x)
        x_relu = self.relu(x_relu)
        # ------------------------------------------
        x_leaky_relu = self.leaky_relu_linear(x)
        x_leaky_relu = self.leaky_relu(x_leaky_relu)
        # ------------------------------------------
        x_tanh = self.tanh_linear(x)
        x_tanh = self.tanh(x_tanh)
        # ------------------------------------------
        x_sigmoid = self.sigmoid_linear(x)
        x_sigmoid = self.sigmoid(x_sigmoid)
        # ------------------------------------------
        x_softplus = self.softplus_linear(x)
        x_softplus = self.softplus(x_softplus)
        #-------------------------------------------
        x_mish = self.mish_linear(x)
        x_mish = self.mish(x_mish)
        # ------------------------------------------
        x_selu = self.selu_linear(x)
        x_selu = self.selu(x_selu)
        # ------------------------------------------
        x_silu = self.silu_linear(x)
        x_silu = self.silu(x_silu)
        # ------------------------------------------
        x_elu = self.elu_linear(x)
        x_elu = self.elu(x_elu)
        # ------------------------------------------
        x_gelu = self.gelu_linear(x)
        x_gelu = self.gelu(x_gelu)
        # ------------------------------------------
        x_softshrink = self.softshrink_linear(x)
        x_softshrink = self.softshrink(x_softshrink)
        #-------------------------------------------
        x_softsign = self.softsign_linear(x)
        x_softsign = self.softsign(x_softsign)
        #-------------------------------------------
        x_cat = torch.cat((x_mish,x_tanh,x_sigmoid,x_softplus,x_silu,x_selu,x_elu,x_gelu,x_leaky_relu,x_relu,x_softshrink,x_softsign),dim=1)
        x_cat = self.layer_norm(x_cat)
        return self.kan(x_cat)
        