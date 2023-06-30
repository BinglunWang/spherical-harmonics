import torch

def sh_encoding(xyz: torch.Tensor, l: int) -> torch.Tensor:
    """
    Follow Nvidia tinycudann implementation:
    https://github.com/NVlabs/tiny-cuda-nn/blob/master/include/tiny-cuda-nn/encodings/spherical_harmonics.h
    """
    assert l >= 1 and l <=4, "order not supported"
    
    xyz = xyz.view(-1, 3)
    code = torch.empty((xyz.shape[0], (l)**2), dtype=xyz.dtype, device=xyz.device)
    
    x, y, z = xyz.float().unbind(-1)
    
    x2, y2, z2 = x*x, y*y, z*z
    xy, yz, xz = x*y, y*z, x*z
    
    
    code[:,0] = 0.28209479177387814
    if l <= 1:
        return code
    code[:,1] = float(-0.48860251190291987) * y
    code[:,2] = float(0.48860251190291987) * z
    code[:,3] = float(-0.48860251190291987) * x
    if l <= 2:
        return code
    code[:,4] = float(1.0925484305920792)*xy
    code[:,5] = float(-1.0925484305920792)*yz
    code[:,6] = float(0.94617469575755997)*z2 - float(0.31539156525251999)
    code[:,7] = float(-1.0925484305920792)*xz
    code[:,8] = float(0.54627421529603959)*x2 - float(0.54627421529603959)*y2
    if l <= 3:
        return code
    code[:,9] = float(0.59004358992664352)*y*(float(-3.0)*x2 + y2)
    code[:,10] = float(2.8906114426405538)*xy*z
    code[:,11] = float(0.45704579946446572)*y*(float(1.0) - float(5.0)*z2)
    code[:,12] = float(0.3731763325901154)*z*(float(5.0)*z2 - float(3.0))
    code[:,13] = float(0.45704579946446572)*x*(float(1.0) - float(5.0)*z2)
    code[:,14] = float(1.4453057213202769)*z*(x2 - y2)
    code[:,15] = float(0.59004358992664352)*x*(-x2 + float(3.0)*y2)
    if l <= 4:
        return code

def test():
    xyz = torch.rand(6, 100, 3)
    return sh_encoding(xyz.view(-1, 3), 3).shape

# test()