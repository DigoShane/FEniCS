#This is a working IpOpt file that does minimization using IPOpt
#This is taken from the url https://fenicsproject.discourse.group/t/multiple-uflinequalityconstraint-with-multiple-control-variables/3533/3

from dolfin import *
from dolfin_adjoint import *

mesh = UnitSquareMesh(16,16)

V0 = FiniteElement("CG", mesh.ufl_cell(), 2)
V1 = FiniteElement("CG", mesh.ufl_cell(), 2)
W = FunctionSpace(mesh, V0*V1)  # mixed function space

w = interpolate(Expression(('sin(pi*x[0])*cos(pi*x[1])', 'cos(3*pi*x[0])'), degree=2), W)
u_0, u_1 = split(w)

m01 = Control(w)

# "Random" functional chosen to test the implementation
J = assemble(u_0**2 *dx) + assemble( u_1**2 *dx)
Jhat = ReducedFunctional(J,m01)

param = {"acceptable_tol": 1.0e-3, "maximum_iterations": 10}

volume_constraint0 = UFLInequalityConstraint(-(0.5 - u_0)*dx, m01)
volume_constraint1 = UFLInequalityConstraint(-(0.5 - u_1)*dx, m01)

problem = MinimizationProblem(Jhat, bounds=[(-1, 1)], constraints=[volume_constraint0,volume_constraint1])

solver = IPOPTSolver(problem, parameters=param)
w_opt = solver.solve() # Solve the optimization

u_0_opt, u_1_opt = split(w_opt)
