#This code solves the time dependent Louville's equation.
#iL u=\partial_t u
#iL=(\partial H\partial p)(\partial/\partial x) - (\partial H/\partial x)(\partial/\partial p)
#H=p^2/2+V(x)
#1. V(x)=a0x^4/4-a1x^2/2+fx
#2. V(x)=a0x^2/2-fx
#F=-V', m=1.

from fenics import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import math

T = 5.60            # final time
num_steps = 200     # number of time steps
dt = T / num_steps # time step size

# Create mesh and define function space
nx=50
ny=50
Lx=1000
Ly=1000
mesh = RectangleMesh(Point(-Lx, -Ly), Point(Lx, Ly), nx, ny)#, diagonal="right")
V = FunctionSpace(mesh, 'P', 2)

# Define boundary condition
#u_D = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-x[1]*x[1])',degree=20, t=0)
u_D = Expression('exp(-(x[0]-0.5)*(x[0]-0.5)-x[1]*x[1])',degree=80)
u_0 = Constant(0)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_0, boundary)

# Define initial value
u_n = interpolate(u_D, V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

#F = Expression('-1*x[0]*x[0]*x[0]+5*x[0]+2',degree=3)#quartic
F = Expression('-1*x[0]+0',degree=3)#quadratic
p = Expression('x[1]',degree=1)
Pe = as_vector([p,F])

#Defining the weak Louville coeff, v(p,-dV/dx)
def L(v):
 return v*Pe


#F = u*v*dx + dt*dot(grad(u), L(v))*dx - u_n*v*dx
#a, L = lhs(F), rhs(F)
a = u*v*dx + dt*dot(grad(u), L(v))*dx 
L = u_n*v*dx

# Time-stepping
u = Function(V)
t = 0
for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    solve(a == L, u, bc)

    if t == T-3*dt:
#    Save solution in VTK format
     vtkfile = File("Transport.pvd")
     vtkfile << u

    # Update previous solution
    u_n.assign(u)

u_e = interpolate(u_D, V)
vtkfile = File("Transport_Ini.pvd")
vtkfile << u_e
