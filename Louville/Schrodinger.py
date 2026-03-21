#HEre we solve the Schrodinger eqn.
#u0=exp(-2x^2-2y^2)
#V(x,y)=(x/2)^2+(3y)^2
#i d_tu=-\Delta u+Vu
#u(x,0)=u0


from fenics import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import math


T = 2.0            # final time
num_steps = 20  # number of time steps
dt = T / num_steps # time step size

mesh = RectangleMesh(Point(-5, -5), Point(5, 5), 55, 55)
V = VectorFunctionSpace(mesh, 'Lagrange', 1, dim=2)
# Define initial value
u_0 = Expression(  ( 'exp(-2*(pow(x[0], 2)+pow(x[1], 2)))', '0'), 
degree=1)
u_n = interpolate(u_0, V)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V,(0, 0), boundary)

u = Function(V)
c = Function(V)
v = TestFunction(V)
f = Expression(('(1/4)*pow(x[0], 2) + 9*pow(x[1], 2)', '0'), degree=2)

t=0

#Residual
ReVL = -u[1]*v[0]*dx + (dt/2)*(-dot(grad(u[0]), grad(v[0])) - 
f[0]*u[0]*v[0])*dx
ReHL = -u_n[1]*v[0]*dx + (dt/2)*(f[0]*u_n[0]*v[0] + dot(grad(u_n[0]), 
grad(v[0])))*dx
ImVL = u[0]*v[1]*dx + (dt/2)*(-dot(grad(u[1]), grad(v[1])) - 
f[0]*u[1]*v[1])*dx
ImHL = u_n[0]*v[1]*dx + dt/2*(f[0]*u_n[1]*v[1] + dot(grad(u_n[1]), 
grad(v[1])))*dx

FReal = ReVL - ReHL
FIm = ImVL - ImHL

Fny = FReal + FIm

J = derivative(Fny, u)
for n in range(num_steps):
    # Update current time
    t += dt
    # Compute solution
    solve(Fny == 0, u, bc, J=J)
    # Update previous solution
    u_n.assign(u)
#    if t == T-1*dt:
#     vtkfile = File("Schordinger.pvd")
#     vtkfile << u_n

c = project(abs(u_n), V)
vtkfile = File("Schordinger.pvd")
vtkfile << c

