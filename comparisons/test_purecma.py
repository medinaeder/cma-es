from cma import purecma
from test_functions import Rosenbrock
p = Rosenbrock(100)
es = purecma.fmin(p, 12*[4], 0.2)[1]

