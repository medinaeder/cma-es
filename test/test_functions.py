import numpy as np
# Suite of test functions taken from wikipeda
# https://en.wikipedia.org/wiki/Test_functions_for_optimization

# Functions with a single solution
class Rosenbrock:
    def __init__(self, dim):
        self.dimension = dim
    def __call__(self, x):
        return self.objective(x)
    def objective(self,x):
        f = 0.
        for i in range(self.dimension-1):
            f+=100*(x[i+1]-x[i]**2)**2+(1-x[i])**2            
        return f
    
    def answer(self):
        return np.ones(self.dimension)
 
class Sphere:
    def __init__(self, dim):
        self.dimension = dim
    
    def objective(self,x):
        return np.dot(x,x)
    def answer(self):
        return np.zeros(self.dimension)


class Ackley:
    def __init__(self):
        self.dimension = 2
    
    def objective(self,x):
        return -20*np.exp(-0.2*np.sqrt(0.5*np.dot(x,x)))-np.exp(0.5*(np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1])))
    def answer(self):
        return np.zeros(self.dimension)

class Rastrigrin:
    def __init__(self,dim):
        self.dimension = dim
    def objective(self,x):
        A = 10
        f = A*self.dimension
        for i in range(self.dimension):
            f+= x[i]*x[i] - A * np.cos(2*np.pi*x[i])
    def answer(self):
        return np.zeros(self.dimension)

class Easom:
    def __init__(self):
        self.dimension = 2
    
    def objective(self,x):
        s = x-np.pi
        return -np.cos(x[0])*np.cos(x[1])*np.exp(-np.dot(s,s))
    def answer(self):
        return np.ones(self.dimension)*np.pi


# Functions with multiple solutions        


# Constrained functions

if __name__=="__main__":
    from cmaes import CMAES
    dim = 2
    tol = 1e-5
    for p in range(5):
        print("Dimension: ", dim)
        problem = Rosenbrock(dim)
        x0 = np.ones(dim)
        cm = CMAES(problem, x0,2)
        cm.optimize()
        xstar = cm.best_solution
        e = xstar-problem.answer()
        assert np.dot(e,e) < tol
        dim*=2
