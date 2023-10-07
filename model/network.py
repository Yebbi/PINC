import numpy as np
import torch.nn as nn
import torch
from torch.autograd import grad


def gradient(inputs, outputs):
    d_points = torch.ones_like(outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0][:, -3:]
    return points_grad

class ImplicitNet_PINC(nn.Module):
    def __init__(
        self,
        d_in, # d_in = dimension of the surface (default d_in = 3)
        dims,
        skip_in=(),
        init_type='geo_relu',
        radius_init=1,
        beta=100.,
    ):
        super().__init__()
        dims = [d_in] + dims + [1 + 2*d_in] # output dim  = 1 (SDF) + 2*d_in (two auxiliary variables for approximaing grad SDF)
        self.d_in = d_in
        self.num_layers = len(dims)
        self.skip_in = skip_in
        self.init_type = init_type
        self.p = 100

        for layer in range(0, self.num_layers - 1):

            if layer + 1 in skip_in:
                out_dim = dims[layer + 1] - d_in
            else:
                out_dim = dims[layer + 1]

            lin = nn.Linear(dims[layer], out_dim)
                
            # if true preform geometric initialization
            if self.init_type == 'geo_relu':
                if layer == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[layer]), std=0.00001)
                    torch.nn.init.constant_(lin.bias, -0.1*radius_init)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            setattr(self, "lin" + str(layer), lin)
            
        if beta > 0:
            self.activation = nn.Softplus(beta=beta)

        # vanilla relu
        else:
            self.activation = nn.ReLU()

    def forward(self, input, return_grad=True, return_auggrad=True):

        output = input
        for layer in range(0, self.num_layers - 1):

            lin = getattr(self, "lin" + str(layer))
            
            if layer in self.skip_in:
                output = torch.cat([output, input], -1) / np.sqrt(2)

            output = lin(output)
            
            if layer < self.num_layers - 2:
                output = self.activation(output)
        
        SDF = output[:,0]
        
        if return_grad: # auxiliary variable
            # implemented only for d_in = 3
            grad_f1 = gradient(input, output[:,1:2])
            grad_f2 = gradient(input, output[:,2:3])
            grad_f3 = gradient(input, output[:,3:4])
            A1 = grad_f3[:,1:2] - grad_f2[:,2:3] - input[:,0:1]/3.
            A2 = grad_f1[:,2:3] - grad_f3[:,0:1] - input[:,1:2]/3.
            A3 = grad_f2[:,0:1] - grad_f1[:,1:2] - input[:,2:3]/3.
            grad_SDF  = torch.cat([A1, A2, A3], dim=-1)
           
            ## When using a finite value for p
            #grad_SDF = grad_SDF / (torch.pow(torch.linalg.norm(grad_SDF, dim=1).unsqueeze(1), (self.p-2)/(self.p-1)) + 1E-10)
            ## When p goes to infinity
            grad_SDF = grad_SDF / (torch.linalg.norm(grad_SDF, dim=1, keepdim=True) + 1E-10)
            
        else:
            grad_SDF = None

        if return_auggrad: # the second auxiliary variable (for curl-free term)
            aug_grad = output[:,1+self.d_in: 1+2*self.d_in] / (torch.nn.ReLU()(torch.linalg.norm(output[:,1+self.d_in: 1+2*self.d_in], dim=1, keepdim=True) - 1) + 1)
        else:
            aug_grad = None
        
        return {"SDF_pred": SDF,
                "grad_pred": grad_SDF,
                "auggrad_pred": aug_grad,}
