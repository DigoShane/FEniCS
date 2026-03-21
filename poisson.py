#Poisson eqn in 2D
from fenics import *
import numpy as np

# Create mesh and define function space
mesh = UnitSquareMesh(32, 32)
V = FunctionSpace(mesh, "Lagrange", 2)

# Define boundary condition
u_D = Expression('1+x[0]*x[0]+2*x[1]*x[1]',degree=2)

# boundary is a function that identifies the boundary points.
def boundary(x, on_boundary):
    return on_boundary

# Define Dirichlet boundary using the FEniCS class DirichletBC
bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6)
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
vtkfile = File("poisson.pvd")
vtkfile << u

# Plot solution
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plot(u)
plt.show()

#Compute Error in L^2 norm
error_L2 = errornorm(u_D, u, 'L2')

print (error_L2)


