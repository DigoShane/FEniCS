#General code to solve 1 particle Louville system.
#iL\psi=i\lambda\psi
#iL=(\partial H\partial p)(\partial/\partial x) - (\partial H/\partial x)(\partial/\partial p)
#H=p^2/2+V(x)
#V(x)=a0x^4/4-a1x^2/2+fx
#F=-V'

from fenics import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

# Define mesh, function space
nx=100
ny=100
Lx=10
Ly=10
mesh = RectangleMesh(Point(-Lx, -Ly), Point(Lx, Ly), nx, ny)#, diagonal="right")
V = FunctionSpace(mesh, "Lagrange", 1)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)

#F = Expression('-1*x[0]*x[0]*x[0]+5*x[0]+2',degree=3)#quintic
F = Expression('-1*x[0]+2',degree=3)#quadratic
p = Expression('x[1]',degree=1)
Pe = as_vector([p,F])
#Defining the weak Louville coeff, v(p,-dV/dx)
def L(v):
 return v*Pe
a = dot(L(v),grad(u))*dx

# Assemble stiffness form
A = PETScMatrix()
assemble(a, tensor=A)

#!!xDx!!##computes the smallest eigen values------------------------
solver = SLEPcEigenSolver(A)
solver.parameters["spectrum"] = "smallest magnitude"
#solver.parameters['solver'] = "lapack"
print(solver.parameters.str(True))

N=3 #smallest N eigen values
solver.solve(N)
#solver.solve()
r, c, rx, cx = solver.get_eigenpair(1)
print("eigenvalue: ", r,"+i ",c)

# Initialize function and assign eigenvector
u = Function(V)
u.vector()[:] = rx
file = File("Louville_r.pvd")
file << u
#plt.plot(u,mode="displacement")
u = Function(V)
u.vector()[:] = cx
file = File("Louville_c.pvd")
file << u
#plt.plot(u,mode="displacement")


#!!xDx!!##computest the largest eigen values-----------------------
#!!xDx!!# Create eigensolver
#!!xDx!!eigensolver = SLEPcEigenSolver(A)
#!!xDx!!
#!!xDx!!# Compute all eigenvalues of A x = \lambda x
#!!xDx!!print("Computing eigenvalues. This can take a minute.")
#!!xDx!!N=100 #first N eigen values
#!!xDx!!eigensolver.solve(N)
#!!xDx!!
#!!xDx!!# Extract smallest eigenpair
#!!xDx!!r, c, rx, cx = eigensolver.get_eigenpair(N-1)
#!!xDx!!print("eigenvalue: ", r,"+i ",c)



