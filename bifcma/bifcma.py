import numpy as np 

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
        
        # Weights Including the Negative Portion
        weights = np.array([np.log((self.lam+1)/2.0) - np.log(i+1) for i in range(self.lam)])
        self.mueff = np.sum(weights[:self.mu])**2/np.sum(weights[:self.mu]**2)

        
        self.c1 = 2 / (( N + 1.3)**2 + self.mueff)                      # Learning rate for rank-one update
        self.cmu = min([1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / ((N + 2)**2 + self.mueff)])  # and for rank-mu update
        self.cc = (4 + self.mueff/N) / (N + 4 + 2 * self.mueff/N)       # Time Constant for C cumulation
        self.cs = (self.mueff + 2 ) / ( N + self.mueff + 5)             # Time Cosntant for Sigma Control
         
        self.damps = 1 +  2 *np.max([0, np.sqrt((self.mueff-1)/(N+1))-1])+ self.cs  # damping for sigma, usually close to 1
        
        mueff_minus = np.sum(weights[self.mu:])**2/np.sum(weights[self.mu:]**2)
        amu_minus = 1 + self.c1/self.cmu
        amueff_minus = 1 +  2 *mueff_minus/(self.mueff+2)  
        apd_minus = (1-self.c1-self.cmu)/(N*self.cmu)
        
        scalepos = 1./np.sum(weights[:self.mu])
        scaleminus = np.amin([amu_minus, amueff_minus, apd_minus])/(-1*np.sum(weights[self.mu:]))
        
        self.weights = weights
        self.weights[:self.mu] = scalepos*self.weights[:self.mu]
        self.weights[self.mu:] = scaleminus*self.weights[self.mu:]
        
         
        
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
        self.C_invsqrt = np.identity(self.N)    # Covariance Matrix 

        
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
        self.tol = 1e-18
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
        print("Lambda = %d " %(self.params.lam))
        header = "Iter  Fevals   BestObj    MaxStd   MinStd"
        print(header)  
        count = 0
        
        # TODO: Add expanded stopping criteria
        while self.fc < self.maxfevals:
            ary = self.sample_solve()
            
            if abs(self.best_fitness - self.prev_best) < self.tol:
                # FIXME: This is a bad stopping criteria because we are sampling
                #        We can "easily" sample the same point
                e = self.xs[-1] - self.best_solution
                if np.dot(e,e) < self.tol:
                    print("Fitness: ", self.best_fitness)
                    print("Solution: ", self.best_solution)
                    break
                
            
            self.prev_best = self.best_fitness
            
            self.update_paths()         # CG-like information
        
            self.update_covariance(ary) # This is equivalent to a Quasi-Newton
            self.update_sigma()
            
            # Store Solutions
            self.fs.append(self.best_fitness)
            self.xs.append(self.best_solution)
            count+=1
            print("%5d %5d %8.6e %6.2e %6.2e" %(count,self.fc, self.best_fitness, np.max(self.D), np.min(self.D)))
            
        
    def sample_solve(self):
        ary = []
        arx = []
        fitness = []
        if self.hints:
            # FIXME:    What is the way to store solutions
            #           We are going to have to profile the code for I/O speed
            self.problems
            self.create_interpolant()
        else:
            for i in range(self.params.lam):
                # Is it easier to sample from the covariance matrix
                # Or do the conversion myself??
                z = np.random.randn(self.N)
                y = np.dot(self.B,np.dot(np.diag(self.D),z))
                x = self.xmean + self.sigma*y
                ary.append(y)
                arx.append(x)
                fitness.append(self.problem.objective(x))
                self.fc+=1
            
            arindex = np.argsort(fitness)
            ymean = np.zeros_like(self.xmean)
            # Updating the mean
            for i in range(self.params.mu):
                ymean+=self.params.weights[i]*ary[arindex[i]]

            self.xmean += self.sigma*ymean
            self.ymean = ymean
            self.best_fitness = fitness[arindex[0]]
            self.best_solution = arx[arindex[0]]
            return [ary[arindex[i]] for i in range(self.params.lam)]
        

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
        
        self.ps = (1-cs)*ps + (np.sqrt(cs*(2-cs)*mueff))*np.dot(self.C_invsqrt,self.ymean)
        
        hsig = np.linalg.norm(self.ps) / (np.sqrt(1 - (1-cs)**(2*self.fc/lam)))/chiN < 1.4 + 2/(self.N+1) 
        self.pc *=(1-cc)
        self.pc += hsig * np.sqrt(cc*(2-cc)*mueff) * self.ymean
        self.hsig = hsig

    def update_covariance(self,ary):
        '''
        C^(k+1) = c1a * C^(k) + c1 * Rk-1 + cmu *Rk-mu
        '''
        ###### Parameters ######
        params = self.params
        cc = params.cc
        c1 = params.c1
        cmu = params.cmu
        hsig = self.hsig
        weights = params.weights
        ########################
        c1a = c1*(1-(1-hsig**2) * cc *(2-cc))    # Modify a few constants 
        
        
        self.C*=(1-c1a-cmu)
        
        # Rank-one update
        self.C += c1*np.outer(self.pc, self.pc)

        # Rank-mu update
        for k in range(params.lam):
            yk = ary[k]
            wk = weights[k]
            if wk > 0:
                self.C+=cmu*weights[k]*np.outer(yk,yk)
            else:
                yn = np.dot(self.C_invsqrt, yk)
                wo = wk * self.N/(np.dot(yn,yn))
                self.C+=cmu*wo*np.outer(yk,yk)
                

        # Recompute D, B via eigendecomposition
        # np.linalg.eigh uses lapack _syevd and _heevd 
        # These are divide and coquer algorithms 
    
        # FIXME: Per the arxiv paper
        #        B and D need to be only recomputed after
        #        iteration = np.max([1, np.floor(1/(10*self.N*(c1+cmu)))]) 
        self.D,self.B = np.linalg.eigh(self.C)
        # self.condition_number = D[0]/D[-1]
        self.D **= 0.5 
        # TODO: Add a check to make sure C is SPD 
        # assert np.all(D) > 0  
        self.C_invsqrt = np.dot(self.B, np.dot(np.diag(1./self.D), self.B.T))

        
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
        self.sigma *= np.exp (sigmafactor)
            
    def create_interpolate(self):
        # TODO: What do we want to do here. Nearest Neighbors?
        pass

    def stop(self):
        # Performs the recommended checks
        # TODO:
        # NoEffectAxis
        
        # NoEffectCoord
        
        # ConditionCov > 1e14
        
        # EqualFunValues  a counter for how many times we haven't moved 
        # EqualFunValues > 10 +np.ceil(30*self.N/self.params.lam)

        # Stagnation Median of the most recent 30% is not better than of the first 30%

        # TolXUp far too small initial sigma or divergent behavior

        # Problem Dependent Stoping Criteria
        
        # TolFun

        # TolX
        pass
        


# TODO: Feasible solution
# Boundary Constraints
# Paper suggest two possible solutions
# 1) If the the solution is strictly inside of the domain then resample plus something else
# 2) Repairing- This seems like a pain in the ass
# Why not work with a penalty method for return np.inf when outside of the domain 
def is_feasible():
    pass


if __name__ == "__main__":
    # Test the code 
    from test_functions import Rosenbrock
    import matplotlib.pyplot as plt
    dim = 10
    problem = Rosenbrock(dim)
    x0 = np.ones(dim)*100
    solver = CMAES(problem,x0,1)
    solver.optimize()
    print(solver.best_solution)

    
