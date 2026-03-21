#General code to solve 1D Ginzburg Landau functional
#\int_{\Omega}(1-u^2)^2/2+(K\nabla u)^2+(MA_0u)^2+\int_{\Scr{R}^3}|\curl A-H|^2
#where H||e_3.
#
from fenics import *

# Create mesh and define function space
#mesh = UnitSquareMesh(32)
mesh = IntervalMesh(32,0,1)
V = FunctionSpace(mesh, "Lagrange", 3)

##,## Defining Dirichlet Boundary conditions
##,# Define Dirichlet boundary (x = 0 or x = 1)
##,def Dboundary(x, on_boundary):
##,    tol = 1e-14
##,    return on_boundary and abs(x[0]) < tol or near(x[0], 1,tol)

## Defining Dirichlet Boundary conditions
# Define Dirichlet boundary (x = 0)
def Dboundary(x, on_boundary):
    tol = 1e-14
    return on_boundary and abs(x[0]) < tol 
##,#def Nboundary(x, on_boundary):
##,#    tol = 1e-14
##,#    return on_boundary and near(x[0], 1,tol) < tol 

## Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, Dboundary)

## Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
# Defining f(x)
f = Constant(2)
#f = Expression('x[0]',degree=1)
# Defining g(s)
g = Constant(-1)
#g = Expression('-4*x[1]',degree=1)
# Defining EA(x)
#EA = Expression('x[0]+1',degree=1)
EA = Constant(1)
# Defining the weak form
a = inner(EA*grad(u), grad(v))*dx
L = f*v*dx+EA*g*v*ds

# Compute solution
u = Function(V)
solve(a == L, u, bc)

#Exact solution
uexact = Expression('x[0]-x[0]*x[0]', degree=2)

# Save solution in VTK format
file = File("1Dbar.pvd")
file << u
file = File("1DBarExact.pvd")
file << uexact

# Compute stress
Q = FunctionSpace( mesh, "DG", 2)
udx = project( u.dx(0), Q)
file1 = File("1DbarStress.pvd")
file1 << udx
