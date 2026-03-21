#Setup for this code is in we want to consider high T1 cases. We want to consider the Gutin model with anisotropic energy.
# and theta=0.
#For the energy functional:-
#1. Pi = ( (1-u**2)**2/2 + K1*inner(grad(u), grad(u))+2*a*u*T1*u.dx(0)+(a**2)*u**2+inner( alpha*grad(a), alpha*grad(a) ) )*dx
#   K1=1, T1=1.6, alpha=1.0
# The tolerance increases with the energy density decreasing. Thus the energy density isn't bounded below.
# 
#2. Pi = ( (1-u**2)**2/2 + K1*inner(grad(u), grad(u)) + K2*inner(grad(u), grad(u))**2 + 2*a*u*T1*u.dx(0) + (a**2)*u**2 + inner( alpha*grad(a), alpha*grad(a) ) )*dx
#   K1=1, K2=1, T1=3, alpha=1.0
# THIS HAS CONVERGED with lx=10.

#======================================================================================================
##Things to do:-
#1.
#======================================================================================================
#ISSUES WITH THE CODE:-

import time # timing for performance test.
import datetime
time0 = time.time()

import dolfin
print(f"DOLFIN version: {dolfin.__version__}")
from dolfin import *
import fenics as fe
import numpy as np
import ufl
print(f" UFL version: {ufl.__version__}")
from ufl import tanh, sin, cos, sqrt, atan, conditional, ne, gt, lt, ln
import matplotlib.pyplot as plt
parameters["form_compiler"]["representation"] = "uflacs"

import sys
np.set_printoptions(threshold=sys.maxsize)

#Parameters
lx = float(input("lx? -->"))
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
#H = float(input("External Magnetic field? -->"));
H = float(0.0) # Learning rate.
#tol = float(input("absolute tolerance? --> "))
tol = float(0.000001) # Learning rate.
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
skp = int(input("Skip tolerance increase check? 1 for Yes, 0 for No --> "))
old_tol = float(input("Old tolerance from previous run? --> "))
K1 = 1.0
K2 = 1.0
T1 = 3.0
alpha = 1.0
kappa = max(K1,T1) # this is the max of all the coeffs.
print("kappa = ", kappa)

#Create mesh and define function space
Nx = np.ceil(lx*50/kappa)
Nx = int(2*np.ceil(Nx/2))
tol_prev = 1 # setting this to 1 randomly.
mesh = IntervalMesh(Nx, 0, lx)

#To denote increase of tolerance
c = int(0)

x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 4)#This is for ExtFile


#========================================================================================================================
# Define functions
a = Function(V)
u = Function(V)
a_up = Function(V)
u_up = Function(V)

#Choice 1
#Pi = ( (1-u**2)**2/2 + K1*inner(grad(u), grad(u)) + 2*a*u*T1*u.dx(0) \
#        + (a**2)*u**2 + inner( alpha*grad(a), alpha*grad(a) ) )*dx
#Choice 2
Pi = ( (1-u**2)**2/2 + K1*inner(grad(u), grad(u)) + K2*inner(grad(u), grad(u))**2 + 2*a*u*T1*u.dx(0) \
        + (a**2)*u**2 + inner( alpha*grad(a), alpha*grad(a) ) )*dx

#Defining the gradients for each branch of the Riemann manifold.
Fa = derivative(Pi, a)
Fu = derivative(Pi, u)
#========================================================================================================================


#========================================================================================================================
##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 #Modified normal state
 print("Using modified bulk Normal as initial condition")
 A = interpolate( Expression("sin(2*x[0])", degree=1), V)
 U = interpolate( Expression("1+0.1*sin(x[0])", degree=1), V)
 
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A = Function(V)
 U = Function(V)
 a_in = XDMFFile("Gurtin-SCIsland-0.xdmf")
 a_in.read_checkpoint(A,"a",0)
 u_in = XDMFFile("Gurtin-SCIsland-1.xdmf")
 u_in.read_checkpoint(U,"u",0)
else:
 sys.exit("Not a valid input for read_in.")
#========================================================================================================================

a_up.vector()[:] = A.vector()[:]
u_up.vector()[:] = U.vector()[:]

#========================================================================================================================

file1 = open("output.txt", "w")
L = ["The list of energies and tolerance at each iteraton:\n"]
file1.writelines(L)
file1.writelines(["=======================================================================\n"])
file1.close()

for tt in range(NN):
 a.vector()[:] = a_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]

 Fa_vec = assemble(Fa)
 Fu_vec = assemble(Fu)

 a_up.vector()[:] = a.vector()[:] - gamma*Fa_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]

 tol_test = np.linalg.norm( np.asarray(Fa_vec.get_local()) + np.asarray(Fu_vec.get_local()) )
 ##Choice 1
 #pie1 = assemble( (1/lx)*( (1-u**2)**2/2 +  K1*inner(grad(u), grad(u)) + 2*a*u*T1*u.dx(0)  \
 #       + (a**2)*u**2 + inner( alpha*grad(a), alpha*grad(a) ) )*dx )
 #Choice 2
 pie1 = assemble( (1/lx)*( (1-u**2)**2/2 +  K1*inner(grad(u), grad(u)) + K2*inner(grad(u), grad(u))**2 + 2*a*u*T1*u.dx(0)  \
        + (a**2)*u**2 + inner( alpha*grad(a), alpha*grad(a) ) )*dx )
 # Append mode: adds to the end of the file
 file1 = open("output.txt", "a")
 L = "tol_test = "+str(tol_test)+", energy = "+str(pie1)+"\n"
 file1.write(L)
 file1.close()
 if float(tol_test)  < tol :
  break

 #print(tol_test)
 if tt == 0 and skp == 0 :
  tol_prev = tol_test
 elif tt >0 and tol_test >= tol_prev and skp ==0 :
  c=1
  print("broke loop at tt=", tt)
  break
 elif tt>0 and tol_test < tol_prev and skp ==0 :
  tol_prev = tol_test

if c == 1 and skp ==0 :
 print("The tolerance has increased.")

h = Function(V)
h = project(alpha*a.dx(0))


##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('Gurtin-SCIsland-0.xdmf')
a1a2tu_out.write_checkpoint(a, "a", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-0.pvd") # for paraview. 
pvd_file << a
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-1.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-1.pvd") # for paraview.
pvd_file << u
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-h.xdmf')
a1a2tu_out.write_checkpoint(h, "h", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-h.pvd") # for paraview.
pvd_file << h
a1a2tu_out.close()

#Defining the energy
##Choice 1
#pie = assemble( (1/lx)*( (1-u**2)**2/2 +  K1*inner(grad(u), grad(u)) + 2*a*u*T1*u.dx(0) \
#        + (a**2)*u**2 + inner(alpha*grad(a), alpha*grad(a) ) )*dx )
#Choice 2
pie = assemble( (1/lx)*( (1-u**2)**2/2 +  K1*inner(grad(u), grad(u)) + K2*inner(grad(u), grad(u))**2 + 2*a*u*T1*u.dx(0)  \
       + (a**2)*u**2 + inner( alpha*grad(a), alpha*grad(a) ) )*dx )

print("================output of code========================")
print("gamma = ", gamma)
print("kappa = ", kappa)
print("lx = ", lx)
print("Nx = ", Nx)
print("NN = ", NN)
print("H = ", float(H))
print("read_in = ", read_in)
print("skip tolerance check = ", skp)
print("Energy density = ", pie)
print("tol = ", float(tol_test))
print("old_tol = ", float(old_tol))



c = plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
c = plot(h)
plt.title(r"$h(x)$",fontsize=26)
plt.show()

time1 = time.time()

print(str(datetime.timedelta(seconds=time1-time0)), "sec for code to run")
print("time = ", time1-time0)

