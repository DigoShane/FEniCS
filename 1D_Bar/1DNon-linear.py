#General code to solve 1D Hyperelastic FEM
#u(0)=0
#u(1)=0
from fenics import *
import ufl
import matplotlib.pyplot as plt

print("Hello")

# Create mesh and define function space
#mesh = UnitSquareMesh(32)
mesh = IntervalMesh(32,0,1)
V = FunctionSpace(mesh, "Lagrange", 3)

print("Hello")

## Defining Dirichlet Boundary conditions
# Define Dirichlet boundary (x = 0 or x = 1)
def Dboundary(x, on_boundary):
    tol = 1e-14
    return on_boundary and abs(x[0]) < tol or near(x[0], 1,tol)

## Define minimization problem
u = TrialFunction(V)
v = TestFunction(V)

f = Constant(1)
t = Constant(0)

y = grad(u)
psi = y*y

Pi = psi*dx-inner(f,u)*dx

F = derivative(Pi,u,v)
J = derivative(F,u,du)

## Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, Dboundary)

# Compute solution
solve(F == 0, u, bc, J)

# Save solution in VTK format
#file = File("1Disp.pvd")
#file << u
#plot(u)
#plt.show()
