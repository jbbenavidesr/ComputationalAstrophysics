# Module with the work from Computational Astrophysics course.
import numpy as np
import matplotlib.pyplot as plt


def graph(x_min, x_max, funct, *args, N=1000, plot=True, x_name=r'$x$', y_name=r'$f(x)$', title = r'Plot of the function $f(x)$',filename='', polar=False, plt_style="seaborn-colorblind",**kwargs):
    """
    Recive a function and graph it over a the given domain of x

    Given that this process is very repetitive, It is easier to have a function that automates the process
    of ploting. Additionally, it returns a tuple with numpy arrays of the plotted information.

    @param x_min: lower bound for x domain
    @param x_max: upper bound for x domain
    @param function: function to be plotted
    @param *args: extra parameters for funct
    
    @param N: number of points on x domain to be calculated
    @param plot: This tells if you want to plot or just get the data.
    @param x_name: str name of x axis
    @param y_name: str name of y axis
    @param title: title to give to the plot.
    @param filename: if given, the plot is saved with that name. 
    @param polar: This allows for the posibility of the graph to be in polar coordinates.
    @param plt_style: Specify the plotting style you want to use. 
    @param **kwargs: extra arguments for the matplotlib call.
    
    @return x_list, y_list: numpy arrays with the plotted data.
    """
    if not callable(funct):
        raise TypeError("funct must be a Python function")

    x_list = np.linspace(x_min, x_max, N, endpoint=False)
    y_list = funct(x_list, *args)
    
    if plot:
        plt.style.use(plt_style)
        
        if not polar:
            plt.plot(x_list, funct(x_list, *args))
            plt.xlabel(x_name)
            plt.ylabel(y_name)
            plt.title(title)
        else:
            plt.polar(x_list, funct(x_list, *args))


        if bool(filename):
            plt.savefig(filename)

        plt.show()
    
    return x_list, y_list

class LagrangeInterpolation:
    """
    This class defines an interpolation through lagrange polinomials of a given set of N data points x, f.
    
    It interpolates a single polinomial of order N-1.
    """
    
    def __init__(self, X, f):
        
        if len(X) != len(f):
            raise ValueError("The number of points in the x_list doesn't correspond to the function points in f_list")

        if len(X) < 2:
            raise ValueError("There must be at least 2 data points to interpolate.")
        
        self.X = X
        self.f = f
        self.N = len(X)
        self.n = self.N-1
        
        self.coeffs = np.empty(self.N, dtype=object)
        for j in range(self.N):
            self.coeffs[j] = self.coeff(j)
        
        
    def coeff(self, j):
        X_j = self.X[j]
    
        def l_coeff(x):
            L = 1
            for k in range(self.N):
                if k != j:
                    X_k = self.X[k]
                    L *= (x - X_k) / (X_j - X_k)
            return L

        return l_coeff
    
    def poli(self, x):
        p = 0
        for j in range(self.N):
            p += self.f[j]*self.coeffs[j](x)

        return p


def sample_f(x_min, x_max, N, funct, *args, **kwargs):
    """
    -----------------------------------------
    Sample function
    -----------------------------------------
    Get an equally spaced sample of a given function
    in a given interval.

    @param x_min, x_max: Limits for the sample to be taken.
    @param N: number of points to be sampled.
    @param func: Python callable that defines the function to be
                sampled. The first argument should be the ordinate
                variable to be sampled from.
    Args and kwargs will be passed to the function after x.
    
    @return x, y: list with the sample points.
    """
    x = np.linspace(x_min, x_max, N)
    y = np.array([funct(i, *args, **kwargs) for i in x])
    
    return x, y

class HermitePolinomial:
    """
    This function is  for getting the polinomial for each couple of points. It asumes that the parameters
    are arrays with 2 points and calculate the polinomial between those.
    """
    def __init__(self, X, f, df):
        
        self.X = X
        self.f = f
        self.df = df
    
    def psi(self, z):
        psi = np.zeros(2)
        psi[0] = 2 * z**3 - 3 * z**2 + 1
        psi[1] = z**3 - 2 * z**2 + z
        
        return psi
        
        
    def z(self, x):
        return (x - self.X[0]) / (self.X[1] - self.X[0])
        
    def p(self, x):
        z = self.z(x)
        ps_z = self.psi(z)
        ps_zm1 = self.psi(1 - z)
        
        p = self.f[0]*ps_z[0] + self.f[1]*ps_zm1[0] + (self.X[1] - self.X[0] )*(self.df[0]*ps_z[1] - self.df[1]*ps_zm1[1]) 
        
        return p

def derivative(x, f):
    '''
    ------------------------------------------
    Derivative(x, f)
    ------------------------------------------
    This function returns the numerical 
    derivative of a discretely-sample function 
    using one-side derivatives in the extreme 
    points of the interval and second order 
    accurate derivative in the middle points.
    The data points may be evenly or unevenly
    spaced.
    ------------------------------------------
    '''
    # Number of points
    N = len(x)
    dfdx = np.zeros([N, 2])
    dfdx[:,0] = x
    
    # Derivative at the extreme points
    dfdx[0,1] = (f[1] - f[0])/(x[1] - x[0])
    dfdx[N-1,1] = (f[N-1] - f[N-2])/(x[N-1] - x[N-2])
    
    #Derivative at the middle points
    for i in range(1,N-1):
        h1 = x[i] - x[i-1]
        h2 = x[i+1] - x[i]
        dfdx[i,1] = h1*f[i+1]/(h2*(h1+h2)) - (h1-h2)*f[i]/(h1*h2) -\
                    h2*f[i-1]/(h1*(h1+h2))
    
    return dfdx

def hermite_interp(X, f):
    if len(X) != len(f):
        raise ValueError("The number of points in the x_list doesn't correspond to the function points in f_list")

    if len(X) < 2:
        raise ValueError("There must be at least 2 data points to interpolate.")
        
    N = len(X)
    df = derivative(X, f)[:, 1]
    
    polis = np.empty(N-1, dtype=object)
    for i in range(N-1):
        xi = X[i:i+2]
        fi = f[i:i+2]
        dfi = df[i:i+2]
        herms = HermitePolinomial(xi, fi, dfi)
        polis[i] = herms.p
    
    def interp(x):
        if x > X[N-1]:
            raise ValueError("x should be least than the maximum value of X")
        
        for i in range(N-2, -1, -1):
            if x >= X[i]:
                return polis[i](x)
        raise ValueError("x should be greater than the minimum value of X")
    
    return interp

def piecewise_lagrange(X, f, n=None):
    if len(X) != len(f):
        raise ValueError("The number of points in the x_list doesn't correspond to the function points in f_list")

    if len(X) < 2:
        raise ValueError("There must be at least 2 data points to interpolate.")
        
    N = len(X)
    if not n:
        n = N-1
    
    polis = np.empty(N-n, dtype=object)
    for i in range(N-n):
        xi = X[i:i+n+1]
        fi = f[i:i+n+1]
        lagi = LagrangeInterpolation(xi, fi)
        polis[i] = lagi.poli
    
    def interp(x):
        if x > X[N-1]:
            raise ValueError("x should be least than the maximum value of X")
        
        for i in range(N-n-1, -1, -1):
            if x >= X[i]:
                return polis[i](x)
        raise ValueError("x should be greater than the minimum value of X")
    
    return interp

def diff_forward(x, func, dx=0.01, *args):
    """
    ----------------------------------------------
    First Order Derivative
    ----------------------------------------------
    Calculates numerically the first order 
    derivative of a given function in point x by using a forward
    step first order approximation.
    
    @param func: Python function of the form $f(x)$ with the function to derivate.
    @param x: point in which the second derivative is to be evaluated.
    @param *args: extra parameters for funct
    @param dx: interval to use for the approximation. (Default= 0.01)
    
    @return diff: first order derivative of the function in point x.
    ----------------------------------------------
    """
    if not callable(func):
        raise TypeError("func must be a python function. Remember not to call it.")
        
    dif = (func(x + dx, *args) - func(x, *args)) / dx
    
    return dif

def diff_backward(x, func, dx=0.01, *args):
    """
    ----------------------------------------------
    First Order Derivative
    ----------------------------------------------
    Calculates numerically the first order 
    derivative of a given function in point x by using a backward
    step first order approximation.
    
    @param func: Python function of the form $f(x)$ with the function to derivate.
    @param x: point in which the second derivative is to be evaluated.
    @param *args: extra parameters for funct
    @param dx: interval to use for the approximation. (Default= 0.01)
    
    @return diff: first order derivative of the function in point x.
    ----------------------------------------------
    """
    if not callable(func):
        raise TypeError("func must be a python function. Remember not to call it.")
        
    dif = (func(x, *args) - func(x - dx, *args)) / dx
    
    return dif

def diff_central(x, func, dx=0.01, *args):
    """
    ----------------------------------------------
    First Order Derivative
    ----------------------------------------------
    Calculates numerically the first order 
    derivative of a given function in point x by using a central
    step second order approximation.
    
    @param func: Python function of the form $f(x)$ with the function to derivate.
    @param x: point in which the second derivative is to be evaluated.
    @param *args: extra parameters for funct
    @param dx: interval to use for the approximation. (Default= 0.01)
    
    @return diff: first order derivative of the function in point x.
    ----------------------------------------------
    """
    if not callable(func):
        raise TypeError("func must be a python function. Remember not to call it.")
        
    dif = (func(x + dx, *args) - func(x - dx, *args)) / (2 * dx)
    
    return dif

def diff_2(x, func, dx=0.01, *args):
    """
    ----------------------------------------------
    Second Order Derivative
    ----------------------------------------------
    Calculates numerically the second order 
    derivative of a given function in point x with
    a second order Taylor approximation.
    
    @param func: Python function of the form $f(x)$ with the function to derivate.
    @param x: point in which the second derivative is to be evaluated.
    @param *args: extra parameters for funct
    @param dx: interval to use for the approximation. (Default= 0.01)
    
    @return diff2: Second order derivative of th e function in point x.
    ----------------------------------------------
    """
    if not callable(func):
        raise TypeError("func must be a python function. Remember not to call it.")
        
    dif = (func(x + dx, *args) + func(x - dx, *args) - 2 * func(x, *args)) / dx**2
    
    return dif

# Esta se puede generalizar a cualquier función.
def derivate(func, *args ,diff= "central", x_min= 0, x_max = 100, dx=0.01):
    """
    ------------------------------
    Function derivative
    ------------------------------
    Calculates the derivative of a function 
    for given values of mean and standard deviation in 
    an interval from x_min to x_max. 
    
    @param func: function to derivate
    @param *args: arguments of funct. The first argument must be the one in which you want to take de derivative.
    @param diff: differentiation algorithm to be used. ("forward", "central", "backward", "2")
    @param x_min, x_max: x range to get the function
    
    @return x, dfunct: Arrays with the x values and the calculated derivative
    ------------------------------
    """
    if not callable(func):
        raise TypeError("funct must be a Python function")
        
    diff = globals()[f"diff_{diff}"]
    
    return graph(x_min, x_max, diff, func, dx, *args, plot=False)