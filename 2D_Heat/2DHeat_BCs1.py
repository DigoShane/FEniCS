#Poisson eqn in 2D.
#The idea here to define different BC's for different portions.

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
##, # Meshing a rectangle with a corner circular hole 
L = 3
R = 1
domain = Rectangle(Point(0,0), Point(L,L)) - Circle(Point(0,0), R)
mesh = generate_mesh(domain, 32)

# Define function space
V = FunctionSpace(mesh, "Lagrange", 2)

# Define boundary condition
class Top(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[1],L) and on_boundary

class Left(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[0],0) and on_boundary

class Bottom(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[1],0) and on_boundary

facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
Top().mark(facets, 1)
Left().mark(facets, 2)
Bottom().mark(facets, 3)
ds = Measure('ds', subdomain_data=facets)

# Define Dirichlet boundary using the FEniCS class DirichletBC
bc = [DirichletBC(V.sub(0), Constant(0), facets, 2), DirichletBC(V.sub(1), Constant(0), facets, 3)]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0)
T_top = Constant((0,1e-3))
a = inner(grad(u), grad(v))*dx
L = inner(f,v)*dx + dot(T,v)*ds(1)

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
vtkfile = File("2DHeat_BC.pvd")
vtkfile << u


