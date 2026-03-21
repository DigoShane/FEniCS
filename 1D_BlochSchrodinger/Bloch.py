#General code to solve 1D Bloch Schrodinger probelm in fenics
#H_k\psi_{nk}=E_n(k)\psi_nk
#H=(p+ik)^2/2m+V(\lambda)
#V(\lambda,x) = 3+A(\lmabda)sin(2\pix/L)
#V(\lambda,x) = 3+cos(2\pi 10x/L-10\lambda)
#a(u,v)=\int \nabla u.\nabla v dx

from fenics import *
#from dolfin import *
import numpy as np

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

# Define mesh, function space
#mesh = Mesh("box_with_dent.xml.gz")
L=5
k = 0.1*2*np.pi/l#Initializing k values
Al = 0.8414709848
mesh = UnitIntervalMesh(10)
mesh = IntervalMesh(10,0,l)
V = VectorFunctionSpace(mesh, "Lagrange", 1, 2)
u = TrialFunction(V)
v = TestFunction(V)
W = as_matrix([[0,-1],[1,0]])
Vpot = Expression('3+al*sin(2*pi*x[0]/l)', pi=np.pi, al=Al, l=l, degree=20)


#Weak form
def kov(v):
 return as_tensor([[s[0], s[2]],
                     [s[2], s[1]]])
#a = 0.5*dot(u.dx[0], v.dx[0])*dx-k*dot(v,dot(W,u.dx[0]))*dx+0.5*k*k*dot(v,u)*dx-dot(v,Vpot*u)*dx
a = 0.5*dot(grad(u), grad(v))*dx
b = dot(v,u)*dx

# Assemble stiffness form
A = PETScMatrix()
B = PETScMatrix()
assemble(a, tensor=A)
assemble(b, tensor=B)

# Create eigensolver
eigensolver = SLEPcEigenSolver(A,B)
solver.parameters["solver"] = "krylov-schur"

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
N=10
eigensolver.solve(N)

for i in range(0,min(N, solver.get_number_converged())):

    # The smallest eigenpair is the first (0-th) one:
    r, c, rx, cx = eigensolver.get_eigenpair(i)
    
    # Turn the eigenvector into a Function:
    rx_func = Function(V)
    rx_func.vector()[:] = rx

    print("eigenvalue: ", r)

    #Initialize function and assign eigenvector
    u = Function(V)
    u.vector()[:] = rx
    
    file = File("Eign/Eigen%s.pvd" % i)
    file << u
