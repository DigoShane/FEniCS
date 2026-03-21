#Poisson eqn in 2D.
#The idea heere to define different geometries.

from dolfin import *
from mshr import *
from fenics import *
import numpy as np
#import matplotlib
#matplotlib.use('agg')
##import matplotlib.pyplot as plt
#import matplotlib.patches as mpatches

## Meshing different domains
##, # Meshing a unit Square
##, mesh = UnitSquareMesh(32, 32)

# Meshing a rectangle with a corner circular hole 
domain = Rectangle(Point(0,0), Point(3,3)) - Circle(Point(0,0), 1)
mesh = generate_mesh(domain, 32)

##,  # Meshing a rectangle with a circular extension 
##,  domain = Rectangle(Point(0,0), Point(3,3)) + Circle(Point(0,0), 1)
##,  mesh = generate_mesh(domain, 32)

#define function space
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
f = Constant(0)
a = inner(grad(u), grad(v))*dx
L = f*v*dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
vtkfile = File("2DHeat_Geo.pvd")
vtkfile << u


