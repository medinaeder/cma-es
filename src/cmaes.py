import numpy as np 

# This is going to be my test function
class Rosenbrock:
    def __init__(self, dim):
        self.dimension = dim
    
    def objective(self,x):
        f = 0.
        for i in range(self.dimension-1):
            f+=100*(x[i+1]-x[i]**2)**2+(1-x[i])**2            
        return f
 
class Sphere:
    def __init__(self, dim):
        self.dimension = dim
    
    def objective(self,x):
        return np.dot(x,x)


class CMAParams:
    def __init__(self, N, popsize = None):
        '''
        N: Optimization Problem Dimension
        popsize: How many offspring we actually want to have
        '''
        if popsize:
            self.lam = int(popsize)
        else:
            self.lam = 4 + int(3 * np.log(N))
            
        self.chiN  = N**0.5 *(1 - 1./(4*N) + 1. / (21 * N **2))

        self.mu  = self.lam//2
        # FIXME:    Does not include negative weights formulation. 
        #           Do we need it?
        weights = [np.log(self.mu + 0.5) - np.log(i+1) for i in range(self.mu)]
        wsum = np.sum(weights)
        self.weights = 1./wsum * np.array(weights)
        self.mueff = wsum/np.dot(self.weights,self.weights)
        
        self.cc = (4 + self.mueff/N) / (N + 4 + 2 * self.mueff/N)       # Time Constant for C cumulation
        self.cs = (self.mueff + 2 ) / ( N + self.mueff + 5)             # Time Cosntant for Sigma Control
        self.c1 = 2 / (( N + 1.3)**2 + self.mueff)                      # Learning rate for rank-one update
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((N + 2)**2 + self.mueff)])  # and for rank-mu update
        self.damps = 2 * self.mueff/self.lam + 0.3 + self.cs  # damping for sigma, usually close to 1
         
        
class CMAES:
    def __init__(self, problem, x0, sigma, popsize = None, hints = False, maxfevals = None):
        self.problem = problem
        self.N = problem.dimension  # Optimization problem dimension
        self.sigma = sigma
        self.xmean = x0
        
        # General Set-Up
        self.params = CMAParams(self.N)

        self.pc = np.zeros(self.N)      # Evolution Path for C
        self.ps = np.zeros(self.N)      # Evolution Path for Sigma
        self.B = np.identity(self.N)    # Rotation Matrix of C
        self.D = np.ones(self.N)        # Diagonal Eigenvalue of C
        self.C = np.identity(self.N)    # Covariance Matrix 

        
        # Offspring inherit solutions as initial_guesses from previous iteration
        self.hints = hints

        # Stopping Criteria
        self.fc = 0     # Number of Function Evaluations
        self.ec = 0     # Number of Eigendecompositions
        
        if maxfevals:
            self.maxfevals = maxfevals
        else:
            if popsize:
                pp = popsize
            else:
                pp = self.params.lam
            self.maxfevals =  100 * pp + 150 * (self.N+3)**2*pp**0.5 
        
        # Change in Objective
        self.tol = 1e-16
        self.prev_best = np.inf # Best Fit

    def optimize(self):
        """ 
            Direct Implementation of CMA-ES 
            Algorithm presented in 
            https://arxiv.org/pdf/1604.00772.pdf
            This will need to be modified

        """ 
        self.fs = []
        self.xs = []
        while self.fc < self.maxfevals:
            arz = self.sample_solve()
            
            if abs(self.best_fitness - self.prev_best) < self.tol:
                print("here")
            
            self.prev_best = self.best_fitness
            
            self.update_paths()         # CG-like information
        
            self.update_covariance(arz) # This is equivalent to a Quasi-Newton
            self.update_sigma()
            
            # Store Solutions
            self.fs.append(self.best_fitness)
            self.xs.append(self.best_solution)
            
        
    def sample_solve(self):
        arz = []
        arx = []
        fitness = []
        if self.hints:
            # FIXME:    What is the way to store solutions
            #           We are going to have to profile the code for I/O speed
            self.problems
            self.create_interpolant()
        else:
            for i in range(self.params.lam):
                z = np.random.randn(self.N)
                x = self.xmean + self.sigma*np.dot(self.B,np.dot(np.diag(self.D),z))
                arz.append(z)
                arx.append(x)
                fitness.append(self.problem.objective(x))
                self.fc+=1
            
            arindex = np.argsort(fitness)
            zmean = np.zeros_like(self.xmean)
            xmean = np.zeros_like(self.xmean)
            # Updating the mean
            for i in range(self.params.mu):
                zmean+=self.params.weights[i]*arz[arindex[i]]
                xmean+=self.params.weights[i]*arx[arindex[i]]

            self.xmean = xmean
            self.zmean = zmean
            self.best_fitness = fitness[arindex[0]]
            self.best_solution = arx[arindex[0]]
            return [arz[arindex[i]] for i in range(self.params.mu)]
        

    def update_paths(self):
        ###### Parameters ######
        params = self.params
        cs = params.cs
        mueff = params.mueff
        lam = params.lam
        cc = params.cc
        chiN = params.chiN
        ########################
        ps = self.ps
        
        self.ps = (1-cs)*ps + (np.sqrt(cs*(2-cs)*mueff))*np.dot(self.B,self.zmean)
        
        # FIXME: Find the exact definition and the reason why this is the case
        hsig = np.linalg.norm(self.ps) / (np.sqrt(1 - (1-cs)**(2*self.fc/lam)))/chiN < 1.4 + 2/(self.N+1) 
        pc = (1-cc)*self.pc + hsig * np.sqrt(cc*(2-cc)*mueff) * np.dot(self.B,np.dot(np.diag(self.D),self.zmean))
        self.pc = pc
        self.hsig = hsig

    def update_covariance(self,arz):
        '''
        C^(k+1) = c1a * C^(k) + c1 * Rk-1 + cmu *Rk-mu
        '''
        ###### Parameters ######
        params = self.params
        cc = params.cc
        c1 = params.c1
        cmu = params.cmu
        hsig = self.hsig
        ########################
        c1a = c1*(1-(1-hsig**2) * cc *(2-cc))    # Modify a few constants 
        
        
        self.C*=c1a
        
        # Rank-one update
        self.C += c1*np.outer(self.pc, self.pc)

        # Rank-mu update
        for k in range(params.mu):
            yk = np.dot(self.B, np.dot(self.B, arz[k]))
            self.C+=cmu*np.outer(yk,yk)

        # TODO: Add a check to make sure C is SPD 
        # Recompute D, B via eigendecomposition
        self.D,self.B = np.linalg.eigh(self.C)
        self.D **= 0.5 
        # FIXME: No stopping criteria based on eigendecomposition count
        self.ec +=1  

    def update_sigma(self):
        ###### Parameters ######
        params = self.params
        cn = params.cs/params.damps
        N = self.N
        ########################
        ps_n = np.dot(self.ps,self.ps)
        sigmafactor = min([1, 0.5*cn*(ps_n/N - 1)])
        print("SF", sigmafactor)
        self.sigma *= np.exp (sigmafactor)
            
    def create_interpolate(self):
        # TODO: What do we want to do here. Nearest Neighbors?
        pass


if __name__ == "__main__":
    # Test the code 
    import matplotlib.pyplot as plt
    dim = 4
    problem = Rosenbrock(4)
    x0 = np.ones(4)*4
    solver = CMAES(problem,x0,3)
    solver.optimize()
    print(solver.best_solution)

    
