#Setup for this code is in we want to consider high T1 cases. We want to consider the Gutin model with anisotropic energy.
# and theta=0.
#We fix the form of u(x)= u0 + u1*eps*sin(x.d/eps) and minimize over A(x) only.
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
T11 = 0.6
T12 = 0.0
T21 = 0.0
T22 = 1.0
u0 = 1.0
u1 = 0.1
eps = 0.1
kappa = max(K11,K12,K22,T11,T12,T21,T22) # this is the max of all the coeffs.
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
Te = as_matrix([[T11,T12],[T21,T22]])

#========================================================================================================================
# Define functions
a1 = Function(V)
a2 = Function(V)
u = Function(V)
u = interpolate( Expression("u0+u1*eps*sin(x[0]/eps)",u0=u0,u1=u1,eps=eps, degree=1), V)
a1_up = Function(V)
a2_up = Function(V)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

#Defining the energy
Pi = ( (1-u**2)**2/2 + inner(Ke*grad(u), grad(u)) + inner(grad(u), grad(u))*inner(grad(u), grad(u)) + 2*inner(as_vector((a1,a2))*u , Te*grad(u)) \
        + (a1**2+a2**2)*u**2 + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx


#Defining the gradients for each branch of the Riemann manifold.
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
#========================================================================================================================


#========================================================================================================================
##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 ##SC state
 #print("Using bulk SC as initial condition")
 #A1 = interpolate( Expression("0.0", degree=1), V)
 #A2 = interpolate( Expression("0.0", degree=1), V)
 #T = interpolate( Expression("1.0", degree=1), V)
 #U = interpolate( Expression("1.0", degree=1), V)
 #
 #Modified normal state
 print("Using modified bulk Normal as initial condition")
 A1 = interpolate( Expression("sin(2*x[0])", degree=1), V)
 A2 = interpolate( Expression("sin(2*x[1])", degree=1), V)
 
 ##Vortex Solution.
 #print("Using Vortex solution")
 #A1 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? -x[1] : \
 #                            -exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
 #                             *x[1]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
 #                               lx=lx, ly=ly, r=0.3517, K=kappa, degree=1), V)
 #A2 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? x[0] : \
 #                            exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
 #                             *x[0]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
 #                               lx=lx, ly=ly, r=0.3517, K=kappa, degree=1), V)
 #### !!xDx!! atan2(f1,f2) = atan(f1/f2)
 #T = interpolate( Expression('atan2(-x[1]+0.5*ly,-x[0]+0.5*lx)+pie',pie=np.pi, lx=lx, ly=ly, degree=1), V)
 #U = interpolate( Expression('tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)))', lx=lx, ly=ly, degree=1), V)
 ##
 ##SC island initial condition.
 #print("Using SC island initial condition")
 #A1 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? -x[1] : \
 #                            -exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
 #                             *x[1]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
 #                               lx=lx, ly=ly, r=0.3517, K=kappa, degree=1), V)
 #A2 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? x[0] : \
 #                            exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
 #                             *x[0]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
 #                               lx=lx, ly=ly, r=0.3517, K=kappa, degree=1), V)
 #U = interpolate( Expression('1', degree=1), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 a1_in = XDMFFile("Gurtin-SCIsland-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in = XDMFFile("Gurtin-SCIsland-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
else:
 sys.exit("Not a valid input for read_in.")
#========================================================================================================================

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]

#========================================================================================================================

file1 = open("output.txt", "w")
L = ["The list of energies and tolerance at each iteraton:\n"]
file1.writelines(L)
file1.writelines(["=======================================================================\n"])
file1.close()

for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]

 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)

 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]

 tol_test = np.linalg.norm( np.asarray(Fa1_vec.get_local()) + np.asarray(Fa2_vec.get_local()) )
 pie1 = assemble( (1/(lx*ly))*( (1-u**2)**2/2 + inner(Ke*grad(u), grad(u)) + inner(grad(u), grad(u))*inner(grad(u), grad(u)) + 2*inner(as_vector((a1,a2))*u , Te*grad(u)) \
        + (a1**2+a2**2)*u**2 + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )
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

h = Function(V)
h = project(curl(a1,a2))


##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('Gurtin-SCIsland-0.xdmf')
a1a2tu_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-0.pvd") # for paraview. 
pvd_file << a1
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-1.xdmf')
a1a2tu_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-1.pvd") # for paraview. 
pvd_file << a2
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-h.xdmf')
a1a2tu_out.write_checkpoint(h, "h", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-h.pvd") # for paraview.
pvd_file << h
a1a2tu_out.close()

#Defining the energy
pie = assemble( (1/(lx*ly))*( (1-u**2)**2/2 + inner(Ke*grad(u), grad(u)) + inner(grad(u), grad(u))*inner(grad(u), grad(u)) + 2*inner(as_vector((a1,a2))*u , Te*grad(u)) \
        + (a1**2+a2**2)*u**2 + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )

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
print("Note that the tensors K, T and M that we specify at the beginning are not all used to determine the energy.")
print("we use +nabla u.Knabla for the enreryg and not K. Then we use nabla u^4 to stabilize.")
print("We are fixing u(x) as a 2-scale function and minimizing over A(x).")
print("We need to vary over u0, u1, eps and d.")
print("================ Note ========================")



c = plot(u)
plt.title(r"$u(x)$",fontsize=26)
cb = plt.colorbar(c)
plt.show()
c = plot(h)
plt.title(r"$h(x)$",fontsize=26)
cb = plt.colorbar(c)
plt.show()

time1 = time.time()

print(str(datetime.timedelta(seconds=time1-time0)), "sec for code to run")
print("time = ", time1-time0)

