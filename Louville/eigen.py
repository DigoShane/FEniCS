#General code to solve 1D eigen value problem in fenics
#Ax=\lambda x
#a(u,v)=\int \nabla u.\nabla v dx

from fenics import *

# Test for PETSc and SLEPc
if not has_linear_algebra_backend("PETSc"):
    print("DOLFIN has not been configured with PETSc. Exiting.")
    exit()

if not has_slepc():
    print("DOLFIN has not been configured with SLEPc. Exiting.")
    exit()

# Define mesh, function space
#mesh = Mesh("box_with_dent.xml.gz")
L=10
nx=50
mesh = RectangleMesh(Point(-L, -L), Point(L, L), nx, nx)#, diagonal="right")
V = FunctionSpace(mesh, "Lagrange", 1)

# Define basis and bilinear form
u = TrialFunction(V)
v = TestFunction(V)
a = dot(grad(u), grad(v))*dx

# Assemble stiffness form
A = PETScMatrix()
assemble(a, tensor=A)

# Create eigensolver
eigensolver = SLEPcEigenSolver(A)

# Compute all eigenvalues of A x = \lambda x
print("Computing eigenvalues. This can take a minute.")
N=50
eigensolver.solve(N)

for i in range(0,N):

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
