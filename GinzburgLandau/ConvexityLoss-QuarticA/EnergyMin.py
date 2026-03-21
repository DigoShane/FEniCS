#Setup for this code is in we want to consider high T1 cases. We want to consider the Gutin model with anisotropic energy.
# and theta=0.
#We want to minimize the following model:
# Pi = \int ( (1-u^2)^2/2 + nabla u.K.nabla u  + A\cdot MA u^2 + u^4/2C_{ijkl}A_iA_jA_kA_l + |\curl A - H|^2 ) dx
# M is symmetric and will lose positive definiteness along one of its eigen values.
# C is permutation invariant. The quartic form in A can be written as
# Phi(A) = alpha1*(A1^4 + A2^4) + alpha2*A3^4 + 2*beta*A1^2*A2^2 + 2*gamma*(A1^2*A3^2 + A2^2*A3^2)
# m(A) =  m1*A1^2 + m1*A2^2 + m3*A3^2
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
ly = float(input("ly? -->"))
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
#H = float(input("External Magnetic field? -->"));
H = float(0.0) # Learning rate.
#tol = float(input("absolute tolerance? --> "))
tol = float(0.000001) # Learning rate.
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
skp = int(input("Skip tolerance increase check? 1 for Yes, 0 for No --> "))
old_tol = float(input("Old tolerance from previous run? --> "))
K11 = 1.0
K12 = 0.0
K22 = 1.0
M11 = 1.0
M22 = 1.0
M33 = -1.0
alpha1 = 2.0
alpha2 = 1.0
beta = 1.0
gammaM = 0.5 # The suffix M is to distinguish from the gamma learning rate.
kappa = max(K11,K12,K22,M11,M22,M33,alpha1,alpha2,beta,gammaM) # this is the max of all the coeffs.
print("kappa = ", kappa)

#Create mesh and define function space
Nx = np.ceil(lx*20/kappa)
Ny = np.ceil(ly*20/kappa)
Nx = int(2*np.ceil(Nx/2))
Ny = int(2*np.ceil(Ny/2))
tol_prev = 1 # setting this to 1 randomly.
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), Nx, Ny) 

#To denote increase of tolerance
c = int(0)

x = SpatialCoordinate(mesh)
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2
V = FunctionSpace(mesh, "Lagrange", 4)#This is for ExtFile

Ke = as_matrix([[K11,K12],[K12,K22]])
Me = as_matrix([[M11,0],[0,M22]])

#========================================================================================================================
# Define functions
a1 = Function(V)
a2 = Function(V)
a3 = Function(V)
u = Function(V)
a1_up = Function(V)
a2_up = Function(V)
a3_up = Function(V)
u_up = Function(V)

def curl1(a1,a2,a3):
    return a3.dx(1) #- a1.dx(2) but d/dx_3=0.

def curl2(a1,a2,a3):
    return -a3.dx(0) # a1.dx(2) but d/dx_3=0.

def curl3(a1,a2,a3):
    return a2.dx(0) - a1.dx(1)

#Defining the energy
#Pi = ( (1-u**2)**2/2 + K11*u.dx(0)**2 + K22*u.dx(1)**2 + (a1**2+a2**2)*u**2 \
#    + 2*T11*u*a1*u.dx(0) + 2*T12*u*a1*u.dx(1) + 2*T21*u*a2*u.dx(0) + 2*T22*u*a2*u.dx(1) \
#      + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

Pi = ( (1-u**2)**2/2 + inner(Ke*grad(u), grad(u)) + (M11*a1**2 + M22*a2**2 +M33*a3**2)*u**2 \
        + (u**4/2)*(alpha1*(a1**4 + a2**4) + alpha2*a3**4 + 2*beta*a1**2*a2**2 \
        + 2*gammaM*(a1**2*a3**2 + a2**2*a3**2)) +  curl1(a1, a2-Ae, a3)*curl1(a1, a2-Ae, a3)\
        + curl2(a1, a2-Ae, a3)*curl2(a1, a2-Ae, a3) + curl3(a1, a2-Ae, a3)*curl3(a1, a2-Ae, a3))*dx


#Defining the gradients for each branch of the Riemann manifold.
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
Fa3 = derivative(Pi, a3)
Fu = derivative(Pi, u)
#========================================================================================================================


#========================================================================================================================
##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 #Modified normal state
 print("Using modified bulk Normal as initial condition")
 A1 = interpolate( Expression("0", degree=1), V)
 A2 = interpolate( Expression("0", degree=1), V)
 A3 = interpolate( Expression("sin(5*x[1])*cos(5*x[0])", degree=1), V)
 U = interpolate( Expression("1", degree=1), V)
 
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 A3 = Function(V)
 U = Function(V)
 a1_in = XDMFFile("Gurtin-ConvLossM-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in = XDMFFile("Gurtin-ConvLossM-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 a3_in = XDMFFile("Gurtin-ConvLossM-2.xdmf")
 a3_in.read_checkpoint(A3,"a3",0)
 u_in = XDMFFile("Gurtin-ConvLossM-3.xdmf")
 u_in.read_checkpoint(U,"u",0)
else:
 sys.exit("Not a valid input for read_in.")
#========================================================================================================================

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
a3_up.vector()[:] = A3.vector()[:]
u_up.vector()[:] = U.vector()[:]

#========================================================================================================================

file1 = open("output.txt", "w")
L = ["The list of energies and tolerance at each iteraton:\n"]
file1.writelines(L)
file1.writelines(["=======================================================================\n"])
file1.close()

for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 a3.vector()[:] = a3_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]

 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 Fa3_vec = assemble(Fa3)
 Fu_vec = assemble(Fu)

 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 a3_up.vector()[:] = a3.vector()[:] - gamma*Fa3_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]

 tol_test = np.linalg.norm( np.asarray(Fa1_vec.get_local()) + np.asarray(Fa2_vec.get_local()) + np.asarray(Fa3_vec.get_local()) + np.asarray(Fu_vec.get_local()) )
 pie1 = assemble( (1/(lx*ly))*( (1-u**2)**2/2 + inner(Ke*grad(u), grad(u)) + (M11*a1**2 + M22*a2**2 +M33*a3**2)*u**2 \
        + (u**4/2)*(alpha1*(a1**4 + a2**4) + alpha2*a3**4 + 2*beta*a1**2*a2**2 \
        + 2*gammaM*(a1**2*a3**2 + a2**2*a3**2)) +  curl1(a1, a2-Ae, a3)*curl1(a1, a2-Ae, a3)\
        + curl2(a1, a2-Ae, a3)*curl2(a1, a2-Ae, a3) + curl3(a1, a2-Ae, a3)*curl3(a1, a2-Ae, a3))*dx )
 # Append mode: adds to the end of the file
 file1 = open("output.txt", "a")
 L = "tol_test = "+str(tol_test)+", energy = "+str(pie1)+"\n"
 file1.write(L)
 file1.close()
 #print("tol_test = ",tol_test, ", energy = ", pie1)
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

h1 = Function(V)
h1 = project(curl1(a1,a2,a3))
h2 = Function(V)
h2 = project(curl2(a1,a2,a3))
h3 = Function(V)
h3 = project(curl3(a1,a2,a3))


##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('Gurtin-ConvLossM-0.xdmf')
a1a2tu_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-ConvLossM-0.pvd") # for paraview. 
pvd_file << a1
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-ConvLossM-1.xdmf')
a1a2tu_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-ConvLossM-1.pvd") # for paraview. 
pvd_file << a2
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-ConvLossM-2.xdmf')
a1a2tu_out.write_checkpoint(a3, "a3", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-ConvLossM-2.pvd") # for paraview. 
pvd_file << a3
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-ConvLossM-3.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-ConvLossM-3.pvd") # for paraview.
pvd_file << u
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-h1.xdmf')
a1a2tu_out.write_checkpoint(h1, "h1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-h1.pvd") # for paraview.
pvd_file << h1
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-h2.xdmf')
a1a2tu_out.write_checkpoint(h2, "h2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-h2.pvd") # for paraview.
pvd_file << h2
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-h3.xdmf')
a1a2tu_out.write_checkpoint(h3, "h3", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-h3.pvd") # for paraview.
pvd_file << h3
a1a2tu_out.close()

#Defining the energy
pie = assemble( (1/(lx*ly))*( (1-u**2)**2/2 + inner(Ke*grad(u), grad(u)) + (M11*a1**2 + M22*a2**2 +M33*a3**2)*u**2 \
        + (u**4/2)*(alpha1*(a1**4 + a2**4) + alpha2*a3**4 + 2*beta*a1**2*a2**2 \
        + 2*gammaM*(a1**2*a3**2 + a2**2*a3**2)) +  curl1(a1, a2-Ae, a3)*curl1(a1, a2-Ae, a3)\
        + curl2(a1, a2-Ae, a3)*curl2(a1, a2-Ae, a3) + curl3(a1, a2-Ae, a3)*curl3(a1, a2-Ae, a3))*dx )

print("================output of code========================")
print("gamma = ", gamma)
print("kappa = ", kappa)
print("lx = ", lx)
print("ly = ", ly)
print("Nx = ", Nx)
print("Ny = ", Ny)
print("NN = ", NN)
print("H = ", float(H))
print("read_in = ", read_in)
print("skip tolerance check = ", skp)
print("Energy density = ", pie)
print("tol = ", float(tol_test))
print("old_tol = ", float(old_tol))
print("================ Note ========================")
print("Please input M -ve definite to get non-unique solutions.")
print("================ Note ========================")



c = plot(u)
plt.title(r"$u(x)$",fontsize=26)
cb = plt.colorbar(c)
plt.show()
c = plot(h1)
plt.title(r"$h1(x)$",fontsize=26)
cb = plt.colorbar(c)
plt.show()
c = plot(h2)
plt.title(r"$h2(x)$",fontsize=26)
cb = plt.colorbar(c)
plt.show()
c = plot(h3)
plt.title(r"$h3(x)$",fontsize=26)
cb = plt.colorbar(c)
plt.show()

time1 = time.time()

print(str(datetime.timedelta(seconds=time1-time0)), "sec for code to run")
print("time = ", time1-time0)

