
from utils import *
import matplotlib.pyplot as plt
from data import t, y


class Kriging:

    def __init__(self, x, y):
        self.b0, self.b1, self.b2, self.g = fit_nelson_siegel(x, y)
        self.x = x
        self.y = y
        self.n = len(x)
        self.smooth = 1.0
        self.gauss = True
        self.t_fit = np.linspace(1, self.n, 500)
        self.h = 1e-5

    def update_smooth(self, smooth):
        self.smooth = smooth

    def update_gauss(self, gauss):
        self.gauss = gauss

    def plot_fns(self):
        y_fit_regression = nelson_siegel(self.t_fit, self.b0, self.b1, self.b2, self.g)
        plt.plot(self.x, self.y, 'o', color='r')
        plt.plot(self.t_fit, y_fit_regression, label='Nelson-Siegel')

    def plot_fk(self):
        y_krigin = kriging_func(self.t_fit, self.x, self.b0, self.b1, self.b2, self.g, self.n, self.smooth, self.gauss, self.y)      
        plt.plot(self.x, self.y, 'o', color='r')
        plt.plot(self.t_fit, y_krigin, label='Kriging')
        
    def plot_f(self):
        self.plot_fns()
        self.plot_fk()
        plt.xlabel('t')
        plt.ylabel('price')

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



k = Kriging(t, y)

#k.update_gauss(False)
k.update_smooth(2.3)

# ------------- Plot kriging ------------- #
k.plot_f()

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

