#This code solves the Transient Heat Transfer problem demonstrated in
#http://home.simula.no/~hpl/homepage/fenics-tutorial/release-1.0-nonabla/webm/timedep.html
#we have set
# u0=0, I= 1+x^2+\alpha y^2+\beta t, f=0

from __future__ import print_function
from fenics import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

T = 4.0            # final time
num_steps = 10     # number of time steps
dt = T / num_steps # time step size
alpha = 3          # parameter alpha
beta = 1.2         # parameter beta

# Create mesh and define function space
nx = ny = 8
mesh = UnitSquareMesh(nx, ny)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + alpha*x[1]*x[1] + beta*t',
                 degree=2, alpha=alpha, beta=beta, t=0)
u_I = interpolate(u_D, V)
vtkfile = File("Transient_Heat_I0.pvd")
vtkfile << u_I

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define initial value
u_n = interpolate(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(beta - 2 - 2*alpha)

F = u*v*dx + dt*dot(grad(u), grad(v))*dx - (u_n + dt*f)*v*dx
a, L = lhs(F), rhs(F)

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt
    u_D.t = t

    # Compute solution
    solve(a == L, u, bc)

#   # Plot solution
#    plot(u)

    if t == T-3*dt:
#    Save solution in VTK format
     vtkfile = File("Transient_Heat.pvd")
     vtkfile << u

    # Update previous solution
    u_n.assign(u)


# Hold plot
#interactive()
