from utils import *
from nelson_siegel_utils import *
from kriging_utils import *
import matplotlib.pyplot as plt
import os
from data import t, y_mid, y_high, y_low

FIG_DIR = 'figures'
for fmt in ['pdf', 'png', 'jpeg', 'eps']:
    os.makedirs(os.path.join(FIG_DIR, fmt), exist_ok=True)



class Kriging:


    # -------------- Initialization ------------- #
    def __init__(self, x, y):
        self.x = x
        self.y = np.array(y)
        self.b0 = 0.0
        self.b1 = 0.0
        self.b2 = 0.0
        self.g = 0.0
        self.b3 = 0.0
        self.g2 = 0.0
        self.n = len(x)
        self.smooth = 1.0
        self.gauss = True
        self.t_fit = np.linspace(1, self.n, 500)
        self.h = 1e-5
        self.penalty_w = 1e6




    # -------------- Parameter updates ------------- #
    def update_smooth(self, smooth):
        self.smooth = smooth

    def update_gauss(self, gauss):
        self.gauss = gauss

    def update_penalty_weight(self, w):
        self.penalty_w = w



    # -------------- Functions for curve-attribute calculations ------------- #
    def kriging(self, x):
        return kriging_func(x, self.x, self.b0, self.b1, self.b2, self.g, self.n, self.smooth, self.gauss, self.y)

    def kriging_svensson(self, x):
        return kriging_svensson_func(x, self.x,  self.b0, self.b1, self.b2, self.b3, self.g, self.g2, self.n, self.smooth, self.gauss, self.y)



    # -------------- Curve fitting ------------- #
    def fit_nelson_siegel(self):
        self.b0, self.b1, self.b2, self.g = fit_nelson_siegel(self.x, self.y)

    def fit_constrained_nelson_siegel(self):
        intervals = list(zip(self.x, y_low, y_high))        
        self.b0, self.b1, self.b2, self.g = constrained_fit_nelson_siegel(self.x, self.y, intervals, self.penalty_w)

    def fit_nelson_siegel_svensson(self):
        self.b0, self.b1, self.b2, self.b3, self.g, self.g2 = fit_nelson_siegel_svensson(self.x, self.y)       

    def fit_constrained_nelson_siegel_svensson(self):
        intervals = list(zip(self.x, y_low, y_high))        
        self.b0, self.b1, self.b2, self.b3, self.g, self.g2 = constrained_fit_nelson_siegel_svensson(self.x, self.y, intervals, self.penalty_w)



    # -------------- Plotting ------------- #     
    def plot_data(self):
        plt.plot(self.x, self.y, 'o', color='b')


    # --- Plot NS & NSS --- #
    def plot_nelson_siegel(self):
        y_fit_regression = nelson_siegel(self.t_fit, self.b0, self.b1, self.b2, self.g)
        plt.plot(self.x, self.y, 'o', color='g', label='liquid price-points')
        plt.plot(self.t_fit, y_fit_regression, color="purple", label='Nelson-Siegel',  alpha=0.7)

    def plot_nelson_siegel_constrained(self):
        y_fit_regression = nelson_siegel(self.t_fit, self.b0, self.b1, self.b2, self.g)
        plt.plot(self.x, self.y, 'o', color='g')
        plt.plot(self.t_fit, y_fit_regression, color="purple", label='Nelson-Siegel (constrained)')
    
    def plot_nelson_siegel_svensson(self):
        y_fit = nelson_siegel_svensson(self.t_fit, self.b0, self.b1, self.b2, self.b3, self.g, self.g2)
        plt.plot(self.x, self.y, 'o', color='r', label='liquid price-points')
        plt.plot(self.t_fit, y_fit, label='Nelson-Siegel-Svensson',  alpha=0.7)

    def plot_nelson_siegel_svensson_constrained(self):
        y_fit_regression = nelson_siegel_svensson(self.t_fit, self.b0, self.b1, self.b2, self.b3, self.g, self.g2)
        plt.plot(self.x, self.y, 'o', color='g')
        plt.plot(self.t_fit, y_fit_regression, color="purple", label='Nelson-Siegel-Svensson (constrained)')
    

     # --- Plot Kriging --- #
    def plot_kriging(self):
        y_fit = kriging_func(self.t_fit, self.x, self.b0, self.b1, self.b2, self.g, self.n, self.smooth, self.gauss, self.y)      
        plt.plot(self.x, self.y, 'o', color='g', label='liquid price-points')
        plt.plot(self.t_fit, y_fit, label='Kriging (NS)', color='blue', alpha=0.7)

    def plot_kriging_svensson(self):
        y_fit = kriging_svensson_func(self.t_fit, self.x, self.b0, self.b1, self.b2, self.b3, self.g, self.g2, self.n, self.smooth, self.gauss, self.y)
        plt.plot(self.x, self.y, 'o', color='r')
        plt.plot(self.t_fit, y_fit, label='Kriging (NS-S)')



    # --- Plot Curve-attributes --- #
    def plot_curve_lenghts(self, f, linspace):
        def f_of_smooth(s):
            self.smooth = s
            return lambda x: np.sqrt(1 + deriv(f, x, self.h)**2)
        plt.plot(linspace, integrate_over_linspace(f_of_smooth, linspace, 1, self.n))


    def plot_curve_smoothness(self, f, linspace):
        def f_of_smooth(s):
            self.smooth = s
            return lambda x: sec_deriv(f, x, self.h) ** 2
        plt.plot(linspace, integrate_over_linspace(f_of_smooth, linspace, 1, self.n))
    




    # -------------- Show ------------- #
    def show(self, title=None, intervals=False, curve_length=False, curve_smooth=False, filename=None):
        if title:
            plt.title(title)
        if intervals:
            cap = 0.1
            for i, (t_i, bid, ask) in enumerate(zip(self.x, y_low, y_high)):
                label = 'Intervals' if i == 0 else None
                alpha = 0.35
                plt.plot([t_i, t_i], [bid, ask], color='black', alpha=alpha, label=label)
                plt.plot([t_i - cap, t_i + cap], [bid, bid], color='black', alpha=alpha)
                plt.plot([t_i - cap, t_i + cap], [ask, ask], color='black', alpha=alpha)
        plt.grid(alpha=0.5)
        if curve_length:
            plt.xlabel('Smoothness Parameter')
            plt.ylabel('Arc Length: ' + r'$\int \sqrt{1 + f\'(x)^{2}} \ dx$')
        elif curve_smooth:
            plt.xlabel('Smoothness Parameter')
            plt.ylabel('Total Curvature: ' + r"$\int f''(x)^{2} \ dx$")
        else:
            plt.xlabel(r'$t$')
            plt.ylabel(r'Price')
            plt.legend()
        if filename:
            for fmt in ['pdf', 'png', 'jpeg', 'eps']:
                plt.savefig(os.path.join(FIG_DIR, fmt, f'{filename}.{fmt}'))
        plt.show()


    def print_nelson_siegel_params(self):
        print(f"B0: {self.b0}")
        print(f"B1: {self.b1}")
        print(f"B2: {self.b2}")
        print(f"g:  {self.g}")
        print(f"B3: {self.b3}")
        print(f"g2: {self.g2}")



    #TODO: Weekly/Monthly SHIFT for smooth prama stability check.




# ---------------------------- EXECUTE ----------------------- #     
k = Kriging(t, y_mid)



# ------------- Update Params. ---------- #
# k.update_gauss(False)
# k.update_smooth(1.9)
# k.update_penalty_weight(10)



# ------------- Plot Data ---------- #
# k.plot_data()
# plt.legend()
# plt.show()





# # ------------- Plot Nelson-Seigel ---------- #
k.fit_nelson_siegel()
k.plot_nelson_siegel()
k.show(filename='ns')

# ------------- Plot Constrained Nelson-Seigel ---------- #
# k.fit_constrained_nelson_siegel()
# k.plot_nelson_siegel_constrained()
# k.show(intervals=True, filename='ns-constrained')





# ------------- Plot Nelson-Seigel-Svensson ---------- #
# k.fit_nelson_siegel_svensson()
# k.plot_nelson_siegel_svensson()
# k.show(filename='ns-svensson')

# ------------- Plot Constrained Nelson-Seigel-Svensson ---------- #
# k.fit_constrained_nelson_siegel_svensson()
# k.plot_nelson_siegel_svensson_constrained()
# k.show(intervals=True, filename='ns-svensson-constrained')



# ------------- Plot Kriging (NS) ------------- #
# k.fit_nelson_siegel()
# k.plot_kriging()
# k.show(filename='kriging')


# ------------- Plot Kriging (Constrained NS) ------------- #
# k.fit_constrained_nelson_siegel()
# k.plot_kriging()
# k.show(filename='kriging-constrained')

# ------------- Plot Kriging (Constrained NS-Svensson) ------------- #
# k.fit_constrained_nelson_siegel_svensson()
# k.plot_kriging_svensson()
# k.show(filename='kriging-svensson-constrained')





# ----------- Plot Curve-length (Kriging NS) ------------ #
# k.fit_nelson_siegel()
# x_smooth = np.linspace(1.0, 3.5, 30) 
# k.plot_curve_lenghts(k.kriging, x_smooth)
# k.show(curve_length=True, filename='arc-length-kriging')


# ----------- Plot Total-curvature (Kriging NS) ------------ #
# k.fit_nelson_siegel()
# x_smooth = np.linspace(0.6, 1.75, 30) 
# k.plot_curve_smoothness(k.kriging, x_smooth)
# k.show(curve_smooth=True, filename='total-curvature-kriging')

