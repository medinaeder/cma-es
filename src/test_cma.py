import numpy as np
import cma

from test_functions import Rosenbrock
import matplotlib.pyplot as plt

es = cma.CMAEvolutionStrategy(np.zeros(8), 0.5)
p = Rosenbrock(8)
while not es.stop():
    sols = es.ask()
    es.tell(sols, [p(sol) for sol in sols])
    #es.logger.add()
    es.disp()

es.result_pretty()
cma.plot()

