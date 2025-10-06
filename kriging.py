from utils import *
import matplotlib.pyplot as plt
from data import t, y
from scipy.integrate import quad

class Kriging:

    def __init__(self, x, y):
        self.b0, self.b1, self.b2, self.g = fit_nelson_siegel(x, y)
        self.x = x
        self.y = y
        self.n = len(x)
        self.smooth = 1.0
        self.gauss = True
        self.t_fit = np.linspace(1, self.n, 500)
        self.h = 1e-8

    def update_smooth(self, smooth):
        self.smooth = smooth

    def update_gauss(self, gauss):
        self.gauss = gauss

    def plot_fns(self):
        y_fit_regression = nelson_siegel(self.t_fit, self.b0, self.b1, self.b2, self.g)
        plt.plot(self.x, self.y, 'o', color='r')
        plt.plot(self.t_fit, y_fit_regression)

    def plot_fk(self):
        y_krigin = kriging_func(self.t_fit, self.x, self.b0, self.b1, self.b2, self.g, self.n, self.smooth, self.gauss, self.y)      
        plt.plot(self.x, self.y, 'o', color='r')
        plt.plot(self.t_fit, y_krigin)
        
    def plot_f(self):
        self.plot_fns()
        self.plot_fk()

    def plot_data(self):
        plt.plot(self.x, self.y, 'o', color='r')

    def plot_fk_curve_lenghts(self, linspace):

        def f_k(x):
            return kriging_func(x, self.x, self.b0, self.b1, self.b2, self.g, self.n, self.smooth, self.gauss, self.y)

        def f_k_integrate(x):
            return np.sqrt(1 + deriv(f_k, x, 0.00001)**2)
        
        plot_curve_length(self, f_k_integrate, linspace, self.n)




k = Kriging(t, y)

k.update_smooth(1.3)
k.plot_f()
plt.show()

x_smooth = np.linspace(1.5, 3, 10)
k.plot_fk_curve_lenghts(x_smooth)
plt.show()
