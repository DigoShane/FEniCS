from dolfin import *
import petsc4py
petsc4py.init()
from petsc4py import PETSc
import numpy as np

## Commets omitted for simplicity: standard introduction
mesh = IntervalMesh(32,0,1)
V = FunctionSpace(mesh, 'P', 3)

def Dboundary(x, on_boundary):
        return on_boundary

## Define boundary condition
u0 = Constant(0.0)
bc = DirichletBC(V, u0, Dboundary)
u = TrialFunction(V)
v = TestFunction(V)
EA = Constant(1)
a = dot(EA*grad(u), grad(v))*dx
f1 = Expression('-x[0]',degree=1)
f2 = Expression('cos(x[0])+sin(x[0])', degree=3)
f3 = Expression('x[0]*x[0]', degree=3)
f4 = Constant(1)
fL = Vector()

fL = [f1, f2, f3, f4]

print("1")

# New suggested trick
A = assemble(a)
bc.apply(A)
solver = LUSolver(A)
#solver.parameters['reuse_factorization'] = True # <- deprecated!

print("2")

# ...and define the solver accordingly
def my_solver(p):
    f = p
    b = assemble(f*v*dx)
    bc.apply(b)
    L = f*v*dx
    u = Function(V)
    solver.solve(u.vector(), b)
    file = File("u4L"+str(i)+".pvd")
    file << u

# Start multiple resolutions
print("Go!")
samples = 4
for i in range(samples):
        my_solver(fL[i])
        print(i)
