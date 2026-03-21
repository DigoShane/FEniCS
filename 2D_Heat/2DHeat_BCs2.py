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
# Meshing a unit Square
mesh = UnitSquareMesh(32, 32)

# Define function space
V = FunctionSpace(mesh, "Lagrange", 2)

# Define boundary condition
class Top(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[1],1) and on_boundary

class Left(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[0],0) and on_boundary

class Bottom(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[1],0) and on_boundary

class Right(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[0],1) and on_boundary

facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
Top().mark(facets, 1)
Left().mark(facets, 2)
Bottom().mark(facets, 3)
ds = Measure('ds', subdomain_data=facets)
n = FacetNormal(mesh)

# Define Dirichlet boundary using the FEniCS class DirichletBC
bc = [DirichletBC(V, Constant(1), facets, 0), DirichletBC(V, Constant(0), facets, 2)]
#bc = [ DirichletBC(V, Constant(0), facets, 2)]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Expression('exp(-pow((x[0] - 1), 2)/(0.5*0.5)-pow((x[1]-1), 2)/(0.5*0.5)-pow((x[2]-1), 2)/(1*1))',degree=3)
q_1= Constant((1,1e-3))
a = inner(grad(u), grad(v))*dx
L = f*v*dx + dot(q_1,n)*v*ds(0)
#L = f*v*dx 

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Save solution in VTK format
vtkfile = File("2DHeat_BC.pvd")
vtkfile << u


