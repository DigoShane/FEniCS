#Here we solve the 1Di gurtin modified Ginzbug Landau problem with an applied magnetic field.
#Here we want to use Energy minimization method. We start off with Gradient Descent.
#HEre a is \ve{A}\cdot e, and u is u. However, the phase \theta=0. But we will keep alpha as a variable
#denoting the angle between the applied the domain and the vector potential.
#------------------------------------------------------------------------------------------------------
# For details on progress, visit the overleaf file:-
#1. The energy functional is that obtained by making the 1D Ansatz that we did in the PRL paper.
# int dx ( (1-u^2)^2/2 + K_1 |u'|^2 + 2T_1uu'A + A^2 u^2 + |sin(\alpha) A'-H|^2 )
# We specifically want T_1^2 > K_1.
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#   b. The relaxation parameter
#   c. The absolute tolerance
#2. When reading from and writing into respective files,
#   we are writing the lagrange multiplier as a constant function
#   When reading the functions, we interpolate onto a space VAu.
#======================================================================================================
#Things to keep in mind about writing this code:-
#1. Define a functoon to evaluate the curl
#2. Define a rotation funciton.
#3. HAve replace L with l throught.
#4. All variables are lower case.
#5. REdo the code by using Hn\cdot B\perp
#6. Implement Nesterov acceleration, momentum, minibatch gradient descent and Noisy Gradient Descent.
#7. put in initial conditions for vortex solution.
#======================================================================================================
#ISSUES WITH THE CODE:-

import dolfin
print(f"DOLFIN version: {dolfin.__version__}")
from dolfin import *
import fenics as fe
import numpy as np
import ufl
print(f" UFL version: {ufl.__version__}")
from ufl import tanh, sin
import matplotlib.pyplot as plt
#import mshr
import sys

#Create mesh and define function space and material parameters.
lx = 10
K1 = Constant(2.0)
T1 = Constant(1.5)
eta = Constant(0.000)
mesh = IntervalMesh(np.ceil(lx*50/T1), 0, lx)
tol_prev = 1 # setting this to 1 randomly.

#To denote increase of tolerance
c = int(0)

x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 4)#This is for ExtFile

# Define functions
a = Function(V)
u = Function(V)
a_up = Function(V)
u_up = Function(V)

# Parameters
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = float(input("External Magnetic field? -->"));
tol = float(input("absolute tolerance? --> "))
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
alpha = float(0.5)

def Stab(u):
    return (u.dx(0)).dx(0)

#Defining the energy
Pi = ( (1-u**2)**2/2 + eta*Stab(u)**2 + K1*inner(grad(u), grad(u)) + 2*T1*u*u.dx(0)*a + a**2*u**2 + inner( alpha*a.dx(0) -H, alpha*a.dx(0) -H ) )*dx

#Defining the gradient
Fa = derivative(Pi, a)
Fu = derivative(Pi, u)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 ##SC state
 #print("Using bulk SC as initial condition")
 #A = interpolate( Expression("0.0", degree=5), V)
 #ALPHA = interpolate( Expression("0.0", degree=2), V)
 #U = interpolate( Expression("1.0", degree=5), V)
 #Modified normal state
 print("Using modified bulk Normal as initial condition")
 A = interpolate( Expression('sin(x[0])', degree=5), V)
 U = interpolate( Expression("x[0]", degree=5), V)
 ##Vortex Solution.
 #..... need to complete
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A = Function(V)
 U = Function(V)
 a_in =  XDMFFile("GL-2DEnrg-0.xdmf")
 a_in.read_checkpoint(A,"a",0)
 u_in =  XDMFFile("GL-2DEnrg-1.xdmf")
 u_in.read_checkpoint(U,"u",0)
else:
 import sys
 sys.exit("Not a valid input for read_in.")

a_up.vector()[:] = A.vector()[:]
u_up.vector()[:] = U.vector()[:]

for tt in range(NN):
 a.vector()[:] = a_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]
 Fa_vec = assemble(Fa)
 Fu_vec = assemble(Fu)
 a_up.vector()[:] = a.vector()[:] - gamma*Fa_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]
 tol_test = np.linalg.norm(np.asarray(Fa_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fu_vec.get_local()))
 ##print(tol_test)
 #if tt == 0 :
 # tol_prev = tol_test
 #else :
 # if tol_test >= tol_prev:
 #  c=1
 #  print("The tolerance has increased.")
 #  print("tolerance =", tol_test)
 #  print("gamma = ", gamma)
 #  print("NN = ", NN)
 #  print("H=", H)
 #  print("read in parameter", read_in)
 #  sys.exit(0)
 # else :
 #  tol_prev = tol_test

 print(tol_test)
 #print(alpha)
 if float(tol_test)  < tol :
  break

##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('GL-2DEnrg-0.xdmf')
a1a2tu_out.write_checkpoint(a, "a", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-0.pvd") # for paraview. 
pvd_file << a
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-1.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-1.pvd") # for paraview.
pvd_file << u
a1a2tu_out.close()


pie = assemble((1/(lx))*((1-u**2)**2/2 + eta*Stab(u)**2 + K1*inner(grad(u), grad(u)) + 2*T1*u*u.dx(0)*a + a**2*u**2 \
                            + inner( alpha*a.dx(0) -H, alpha*a.dx(0) -H ) )*dx )
print("Energy density =", pie)
print("tolerance =", tol_test)
print("gamma = ", gamma)
print("NN = ", NN)
print("H=", H)
print("alpha", alpha)
print("read in parameter", read_in)
print("===========================================")
print("iterate over alpha")
print("===========================================")

h = Function(V)
h = project(a.dx(0)*alpha) 

c = plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
c = plot(a)
plt.title(r"$A(x)$",fontsize=26)
plt.show()
c = plot(h)
plt.title(r"$h(x)$",fontsize=26)
plt.show()

