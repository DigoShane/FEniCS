#  K
## Solving Ax=b with different right hand sides
#  Essentials: matplotlib
	
#=============================================

import numpy as np
from fenics import *
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

mesh = IntervalMesh(100,0,1)
V = FunctionSpace(mesh, 'Lagrange', 1) # you can use 'P' instead of Lagrange
u_D = Expression('0.0', degree=0)
def Dir_boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, Dir_boundary)
u = TrialFunction(V)
v = TestFunction(V)
C_elastic = Expression('1.0', degree=0)


## Set forcing: 
f1 = Expression('-x[0]',degree=1)
f2 = Expression('cos(x[0])+sin(x[0])', degree=1)
f3 = Expression('A*cos(x[0])+B*sin(x[0])- x[0]*x[0]', A=2.0, B=Constant(5.0), degree=1)

f3.A = 5.0

a = -dot(grad(u), grad(v))*dx + C_elastic*u*v*dx
L = f3*v*dx  


# Generating Matrix and vector
A, b = assemble_system(a,L,bc)

aM = -dot(grad(u), grad(v))*dx + C_elastic*u*v*dx

bM = [f1*v*dx, f2*v*dx, f3*v*dx]

## declare a dolfin solution vector 
x=Vector()
xM=Matrix()

## Solver choice
#solve(A,x,b)
solve(A,x,b, "lu")
solve(aM,xM,bM)
#solve(A,x,b, "gmres","ilu")
#solve(A,x,b, "cg","hypre_amg")

## Extract solution for plotting and post processing
X_coord = mesh.coordinates()
xvals = X_coord[:,0]

## Plot solution and mesh

plt.plot(xvals,x,'r',linewidth=2)
plt.grid()
plt.xlabel('x')
plt.ylabel('u(x)')
#plot(mesh)


# Hold plot
plt.savefig('Ax_b.pdf',format='pdf')
plt.show()
