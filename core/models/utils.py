"""
Utility functions for model operations.

Ported from refusal_direction/pipeline/utils/utils.py
"""

import torch
import einops
from torch import Tensor
from jaxtyping import Float


def get_orthogonalized_matrix(
    matrix: Float[Tensor, '... d_model'], 
    vec: Float[Tensor, 'd_model']
) -> Float[Tensor, '... d_model']:
    """
    Orthogonalize a matrix with respect to a vector.
    
    Removes the component of each row of the matrix that lies along the direction of vec.
    Used for weight orthogonalization in ablation experiments.
    
    Args:
        matrix: The matrix to orthogonalize, shape (..., d_model)
        vec: The vector to orthogonalize against, shape (d_model,)
        
    Returns:
        The orthogonalized matrix with the same shape as input
    """
    vec = vec / torch.norm(vec)
    vec = vec.to(matrix)

    proj = einops.einsum(matrix, vec.unsqueeze(-1), '... d_model, d_model single -> ... single') * vec
    return matrix - proj
