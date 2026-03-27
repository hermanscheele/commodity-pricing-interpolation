import random
import numpy as np
from utils import construct_intervals
from nelson_siegel_utils import fit_nelson_siegel, constrained_fit_nelson_siegel, nelson_siegel
from data import y_mid, y_high, y_low, t




y_mid = np.array(y_mid)



def draw_random_interval_vector(I):
    n = len(I)
    v = []

    for i in range(n):
        high = I[i][1]
        low = I[i][2]
        draw = random.uniform(low, high)        
        v.append(draw)

    return v



I = construct_intervals(t, y_high, y_low)






def construct_covariance(x, R, sigma_sq, rho):

    n = len(x)
    Sigma = np.zeros((n, n))
    
    for k in range(n):
        for l in range(n):
            Sigma[k, l] = sigma_sq * R(x[k], x[l], rho)
    
    return Sigma


# Example usage:
def R(x_k, x_l, rho):
    return np.exp(-0.5 * ((x_k - x_l) / rho)**2)

x_i = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
Sigma = construct_covariance(x_i, R, sigma_sq=1.0, rho=0.5)




# Constructing and drawing vector from the gaussian distribution ϕ(z | ẑ^e)
# Same sequential steps as the algorithm from paper

def draw_gaussian_interval_vector(intervalls):

    t_vals = [i[0] for i in intervalls]
    t_vals = np.array(t_vals)

    # Define my_i (THIS IS MAYBE WRONG) (WHAT IS x_i)
    b0_i, b1_i, b2_i, g_i = constrained_fit_nelson_siegel(t_vals, y_mid, intervalls)
    my_i = nelson_siegel(t_vals, b0_i, b1_i, b2_i, g_i)

    # Define Sigma_i
    sigma_i = construct_covariance()



    # Define my_e (THIS IS MAYBE WRONG) (WHAT IS x_e)
    b0_e, b1_e, b2_e, g_e = fit_nelson_siegel(t_vals, y_mid)
    my_e = nelson_siegel(t_vals, b0_e, b1_e, b2_e, g_e)





    


    pass






a = draw_gaussian_interval_vector(I)

print(a)



