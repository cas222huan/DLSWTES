import numpy as np

C1 = 1.19104e8 # W·μm–4·Sr–1·m–2
C2 = 1.43877e4 # μm·K

# calculate spectral radiance for black body (unit: W·μm–1·Sr–1·m–2)
def Planck_law(T, lamda):
    # T: temperature, lamda: wavelength in μm
    R = C1 / lamda**5 / (np.exp(C2/(lamda*T))-1)
    return R

# calculate temperature from spectral radiance
def Inverse_Planck_law(R, lamda):
    # L: spectral radiance, lamda: wavelength in μm
    A = C1 / (R*lamda**5) + 1
    T = C2 / (lamda * np.log(A))
    return T

# class TES_calculator: