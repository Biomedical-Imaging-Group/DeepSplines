import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class LinearSpline(nn.Module):
    """
    Class for LinearSpline activation functions

    Args:
        num_coeffs (int): number of coefficients of the spline (including the ones at the boundary)
        num_activations (int) : number of activation functions
        x_min (float): position of left-most coeff
        x_max (float): position of right-most coeff
        slope_min (float or None): minimum slope of the activation
        slope_max (float or None): maximum slope of the activation
    """

    def __init__(self, num_activations, num_coeffs, x_min, x_max, spline_init, 
                 slope_min=None, slope_max=None, apply_scaling=False):

        super().__init__()

        self.num_activations = int(num_activations)
        self.num_coeffs = int(num_coeffs)
        self.register_buffer("x_min", torch.tensor([x_min]))
        self.register_buffer("x_max", torch.tensor([x_max]))
        self.init = spline_init
        self.slope_min = slope_min
        self.slope_max = slope_max
        self.apply_scaling = apply_scaling
        self.register_buffer("grid_size", (self.x_max - self.x_min) / (self.num_coeffs - 1))
        self.register_buffer("D2_filter", Tensor([1, -2, 1]).view(1, 1, 3).div(self.grid_size))
        
        # parameters
        self.coefficients = nn.Parameter(self.initialize_coeffs())  # spline coefficients
        if (self.apply_scaling):
            self.scaling_factors = nn.Parameter(torch.ones((1, self.num_activations, 1, 1)))

        # Initialize indices of the leftmost coefficient of each activation function.
        self.register_buffer("leftmost_coeff_indices", (torch.arange(0, self.num_activations) * self.num_coeffs))
        

    def initialize_coeffs(self):
        """The coefficients are initialized with the value of the activation
        # at each knot (c[k] = f[k], since linear splines are interpolators)."""
        grid_tensor = torch.linspace(self.x_min.item(), self.x_max.item(), self.num_coeffs)
        grid_tensor = grid_tensor.expand((self.num_activations, self.num_coeffs))

        if isinstance(self.init, float):
            coefficients = torch.ones_like(grid_tensor)*self.init
        elif self.init == 'identity':
            coefficients = grid_tensor
        elif self.init == 'relu':
            coefficients = F.relu(grid_tensor)
        else:
            raise ValueError('init should be in [identity, relu] or a number.')
        
        return coefficients
    

    @property
    def projected_coefficients(self):
        """Projection of B-spline coefficients such that they satisfy the slope constraints"""
        if not (self.slope_min is None and self.slope_max is None):
            coeffs = self.coefficients
            new_slopes = torch.clamp((coeffs[:, 1:] - coeffs[:, :-1]) / self.grid_size, self.slope_min, self.slope_max)
            proj_coeffs = torch.zeros_like(coeffs)
            proj_coeffs[:,1:] = torch.cumsum(new_slopes, dim=1) * self.grid_size
            proj_coeffs = proj_coeffs + torch.mean(coeffs - proj_coeffs, dim=1).unsqueeze(1)
        else:
            proj_coeffs = self.coefficients

        return proj_coeffs

    
    @property
    def relu_slopes(self):
        """ Get the activation relu slopes {a_k},
        by doing a valid convolution of the coefficients {c_k}
        with the second-order finite-difference filter [1,-2,1].
        """
        return F.conv1d(self.projected_coefficients.unsqueeze(1), self.D2_filter).squeeze(1)
    

    def forward(self, x):
        """
        Args:
            input (torch.Tensor):
                2D or 4D, depending on whether the layer is
                convolutional ('conv') or fully-connected ('fc')

        Returns:
            output (torch.Tensor)
        """
        input_size = x.size()
        if len(input_size) == 2:
            # transform to 4D size (N, num_units=num_activations, 1, 1)
            x = x.view(*input_size, 1, 1)
        
        if (self.apply_scaling):
            x = x.mul(self.scaling_factors)
       
        projected_coefficients_vect = self.projected_coefficients.view(-1)
        x = LinearSpline_Func.apply(x, projected_coefficients_vect, self.x_min, self.x_max, self.grid_size, self.leftmost_coeff_indices)
        
        if (self.apply_scaling):
            x = x.div(self.scaling_factors)
        x = x.view(*input_size)

        return x


    def extra_repr(self):
        """ repr for print(model) """

        s = ('num_activations={num_activations}, '
             'init={init}, num_coeffs={num_coeffs}, range=[{x_min[0]:.3f}, {x_max[0]:.3f}], '
             'slope_max={slope_max}, '
             'slope_min={slope_min}, '
             'apply_scaling={apply_scaling}.'
             )

        return s.format(**self.__dict__)
        

    def tv2(self):
        """
        Computes the second-order total-variation regularization.

        deepspline(x) = sum_k [a_k * ReLU(x-kT)] + (b1*x + b0)
        The regularization term applied to this function is:
        TV(2)(deepsline) = ||a||_1.
        """
        return self.relu_slopes.norm(1, dim=1).sum()
    


class LinearSpline_Func(torch.autograd.Function):
    """
    Autograd function to only backpropagate through the B-splines that were
    used to calculate output = activation(input), for each element of the
    input.
    """
    @staticmethod
    def forward(ctx, x, coefficients_vect, x_min, x_max, grid_size, leftmost_coeff_indices):

        # The value of the spline at any x is a combination of at most two coefficients
        x_clamped = x.clamp(min=x_min.item(), max=x_max.item() - grid_size.item())
        left_coeff_idx = torch.floor((x_clamped - x_min) / grid_size)  #left coefficient

        # This gives the indices (in coefficients_vect) of the left coefficients
        indices = (leftmost_coeff_indices.view(1, -1, 1, 1) + left_coeff_idx).long()

        fracs = (x - x_min) / grid_size - left_coeff_idx  # distance to left coefficient

        # Only two B-spline basis functions are required to compute the output
        # (through linear interpolation) for each input in the B-spline range.
        output = coefficients_vect[indices + 1] * fracs + coefficients_vect[indices] * (1 - fracs)

        ctx.save_for_backward(fracs, coefficients_vect, indices, grid_size)
        return output

    @staticmethod
    def backward(ctx, grad_out):

        fracs, coefficients_vect, indices, grid_size = ctx.saved_tensors

        grad_x = (coefficients_vect[indices + 1] - coefficients_vect[indices]) / grid_size * grad_out

        # Next, add the gradients with respect to each coefficient, such that,
        # for each data point, only the gradients wrt to the two closest
        # coefficients are added (since only these can be nonzero).
        grad_coefficients_vect = torch.zeros_like(coefficients_vect, dtype=coefficients_vect.dtype)
        # right coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indices.view(-1) + 1, (fracs * grad_out).view(-1))
        # left coefficients gradients
        grad_coefficients_vect.scatter_add_(0, indices.view(-1), ((1 - fracs) * grad_out).view(-1))

        return grad_x, grad_coefficients_vect, None, None, None, None