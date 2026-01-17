from utils import *
from nelson_siegel_utils import *
from kriging_utils import *
import matplotlib.pyplot as plt
from data import t, y_mid, y_high, y_low


class Kriging:

    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y)
        self.b0, self.b1, self.b2, self.g = fit_nelson_siegel(self.x, self.y)
        self.n = len(x)
        self.smooth = 1.0
        self.gauss = True
        self.t_fit = np.linspace(1, self.n, 500)
        self.h = 1e-5



    def update_smooth(self, smooth):
        self.smooth = smooth


    def update_gauss(self, gauss):
        self.gauss = gauss


    def fit_constrained_nelson_siegel(self):
        intervals = list(zip(self.x, y_low, y_high))        
        self.b0, self.b1, self.b2, self.g = constrained_fit_nelson_siegel(self.x, self.y, intervals)


    def plot_constrained_fns(self):
        y_fit_regression = nelson_siegel(self.t_fit, self.b0, self.b1, self.b2, self.g)
        plt.vlines(t, y_low, y_high, color='gray', alpha=0.4, label="Intervals")
        plt.plot(self.t_fit, y_fit_regression, color="green", label='Nelson-Siegel (constrained)')


    def plot_fns_midpoints(self):
        y_fit_regression = nelson_siegel(self.t_fit, self.b0, self.b1, self.b2, self.g)
        plt.plot(self.t_fit, y_fit_regression, label='Nelson-Siegel')


    def plot_fk(self):
        y_krigin = kriging_func(self.t_fit, self.x, self.b0, self.b1, self.b2, self.g, self.n, self.smooth, self.gauss, self.y)      
        plt.plot(self.x, self.y, 'o', color='r')
        plt.plot(self.t_fit, y_krigin, label='Kriging')
        
    
    def plot_data(self):
        plt.plot(self.x, self.y, 'o', color='r')


    def f_k(self, x):
            return kriging_func(x, self.x, self.b0, self.b1, self.b2, self.g, self.n, self.smooth, self.gauss, self.y)


    def plot_fk_curve_lenghts(self, f_k, linspace):
        
        def f_k_integrate(x):
            return np.sqrt(1 + deriv(f_k, x, self.h)**2)
       
        plot_f_integration(self, f_k_integrate, linspace, self.n)


    def plot_fk_curve_smoothness(self, f_k, linspace):

        def f_k_integrate(x):
            return sec_deriv(f_k, x, self.h) ** 2

        plot_f_integration(self, f_k_integrate, linspace, self.n)
        plt.xlabel('smoothness param.')
        plt.ylabel('total curvature: ' + r'$\int \text{sec.deriv(f)}^{2} \ $')
    

    def print_fns_params(self):
        print(f"B0: {self.b0}")
        print(f"B1: {self.b1}")
        print(f"B2: {self.b2}")
        print(f"g: {self.g}")






k = Kriging(t, y_mid)



#k.update_gauss(False)
k.update_smooth(1.9)


# ------------- Plot Nelson-Seigel ---------- #
k.plot_fns_midpoints()


# ------------- Plot Constrained Nelson-Seigel ---------- #
k.fit_constrained_nelson_siegel()
k.plot_constrained_fns()

plt.legend()
plt.show()


# ------------- Plot kriging ------------- #
k.plot_fk()
plt.grid(True, alpha=0.3)

plt.legend()
plt.savefig("figures/nelson_siegel_kriging.pdf")
plt.show()


# ----------- Plot curvature ------------ #
x_smooth = np.linspace(0.7, 1.3, 10) 
#k.plot_fk_curve_lenghts(k.f_k, x_smooth)
k.plot_fk_curve_smoothness(k.f_k, x_smooth)

plt.grid(True, alpha=0.3)
plt.savefig("figures/curve_smoothness_vs_smooth_param.pdf")
plt.show()

