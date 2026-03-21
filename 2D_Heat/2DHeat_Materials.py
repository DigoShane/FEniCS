################################################################################
#              !!!This doesn't work!!!
################################################################################
#Poisson eqn in 2D.
#The idea here to define different material parameters for different subdomains

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
##,# Meshing a rectangle with a corner circular hole 
##,domain = Rectangle(Point(0,0), Point(3,3)) - Circle(Point(0,0), 1)
##,mesh = generate_mesh(domain, 32)

#define function space
V = FunctionSpace(mesh, "Lagrange", 2)

# Define boundary condition
u_D = Expression('1+x[0]*x[0]+2*x[1]*x[1]',degree=2)

# Defining tolerance
tol = 1e-14
k_0 = 1
k_1 = 2

# boundary is a function that identifies the boundary points.
##,def boundary(x, on_boundary):
##,    return on_boundary
class Boundary(SubDomain):
 def inside(self, x, on_boundary):
  return on_boundary and near(x[0],0,tol)

boundary = Boundary()

# Defining Material parameters
class Omega_0(SubDomain):
 def inside(self, x, on_boundary):
  return x[1] <= 0.5 + tol

class Omega_1(SubDomain):
 def inside(self, x, on_boundary):
  return x[1] >= 0.5 - tol

materials = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)

subdomain_0 = Omega_0()
subdomain_1 = Omega_1()
subdomain_0.mark(materials, 0)
subdomain_1.mark(materials, 1)

File('Materials.xml.gz') << materials

class K(Expression):
 def __init__(self, materials, k_0, k_1, **kwargs):
  self.materials = materials
  self.k_0 = k_0
  self.k_1 = k_1

def eval_cell(self, values, x, cell):
 if self.materials[cell.index] == 0:
  values[0] = self.k_0
 else:
  values[0] = self.k_1

kappa = K(materials, k_0, k_1, degree=0)

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
vtkfile = File("2DHeat_Mat.pvd")
vtkfile << u


