#Explicitly model discontinuities while allowing to refine the core seperately.
#this is testing for solutions where theta is single valued.
#======================================================================================================
##Things to do:-
#1. get rid of theta and all associated solutions. (DONE)
#2. modify folder location and file names. (DONE)
#3. Modify the energy to be anisotropic. (DONE) 
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
from ufl import tanh, sin, cos, sqrt, atan, conditional, ne, gt, lt
import matplotlib.pyplot as plt
parameters["form_compiler"]["representation"] = "uflacs"

import sys
np.set_printoptions(threshold=sys.maxsize)

#Parameters
gamma = float(0.000001) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = Constant(0.3);
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
tol = float(0.000001)

#Create mesh and define function space
tol_prev = 1 # setting this to 1 randomly.
mesh = RectangleMesh(Point(0., 0.), Point(10, 10), 100, 100) 

x = SpatialCoordinate(mesh)
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2
V = FunctionSpace(mesh, "Lagrange", 4)#This is for ExtFile

mesh.init(0,1)

#========================================================================================================================
# Define functions
a1 = Function(V)
a2 = Function(V)
fR = Function(V)
fI = Function(V)
a1_up = Function(V)
a2_up = Function(V)
fR_up = Function(V)
fI_up = Function(V)

def atan_2(y,x):
    return conditional(ne(y, 0), 2*atan((sqrt(x**2+y**2)-x)/y), conditional(gt(x,0), 0, conditional(lt(x,0), pi, 1.0e20)))

def ssqrt(x):
    return sqrt(abs(x)+ 1e-10)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

def uf(fR,fI):
    return ssqrt(fR**2+fI**2)

def theta(fR,fI):
    return project(atan_2(fI,fR), V)

#Defining the energy
Pi = ( (a1-theta(fR,fI).dx(0))*(a1-theta(fR,fI).dx(0)) + (a1-theta(fR,fI).dx(0))*(a2-theta(fR,fI).dx(1)) + (a2-theta(fR,fI).dx(1))*(a2-theta(fR,fI).dx(1)) \
     + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

#Defining the gradients for each branch of the Riemann manifold.
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
FfR = derivative(Pi, fR)
FfI = derivative(Pi, fI)
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
 ##Modified normal state
 #print("Using modified bulk Normal as initial condition")
 #A1 = interpolate( Expression("0.0", degree=1), V)
 #A2 = interpolate( Expression("H*x[0]", H=H, degree=1), V)
 #T = interpolate( Expression("x[1]", degree=1), V)
 #U = interpolate( Expression("x[0]", degree=1), V)
 #
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
 #
 ##SC island initial condition.
 print("Using SC island initial condition")
 A1 = interpolate( Expression('1', degree=1), V)
 A2 = interpolate( Expression('x[0]', degree=1), V)
 FR = interpolate( Expression('sqrt(1-tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)))*tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) )', lx=10, ly=10, degree=1), V)
 FI = interpolate( Expression('1', degree=1), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 FR = Function(V)
 FI = Function(V)
 a1_in = XDMFFile("Gurtin-SCIsland-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in = XDMFFile("Gurtin-SCIsland-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 fR_in = XDMFFile("Gurtin-SCIsland-2.xdmf")
 fR_in.read_checkpoint(FR,"fR",0)
 fI_in = XDMFFile("Gurtin-SCIsland-3.xdmf")
 fI_in.read_checkpoint(FI,"fI",0)
else:
 sys.exit("Not a valid input for read_in.")


#========================================================================================================================

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
fR_up.vector()[:] = FR.vector()[:]
fI_up.vector()[:] = FI.vector()[:]

#========================================================================================================================
for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 fR.vector()[:] = fR_up.vector()[:]
 fI.vector()[:] = fI_up.vector()[:]

 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 FfR_vec = assemble(FfR)
 FfI_vec = assemble(FfI)

 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 fR_up.vector()[:] = fR.vector()[:] - gamma*FfR_vec[:]
 fI_up.vector()[:] = fI.vector()[:] - gamma*FfI_vec[:]

 #print(Fa1_vec.get_local()) # prints the vector.
 #print(np.linalg.norm(np.asarray(Fa1_vec.get_local()))) # prints the vector's norm.
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(FfR_vec.get_local()))\
           +np.linalg.norm(np.asarray(FfI_vec.get_local()))

 print(tol_test)
 if float(tol_test)  < tol :
  break

h = Function(V)
h = project(curl(a1,a2))
u = Function(V)
u = project(uf(fR,fI))

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
a1a2tu_out = XDMFFile('Gurtin-SCIsland-2.xdmf')
a1a2tu_out.write_checkpoint(fR, "fR", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-2.pvd") # for paraview.
pvd_file << fR
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-3.xdmf')
a1a2tu_out.write_checkpoint(fI, "fI", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-3.pvd") # for paraview.
pvd_file << fI
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-h.xdmf')
a1a2tu_out.write_checkpoint(h, "h", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-h.pvd") # for paraview.
pvd_file << h
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-u.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-u.pvd") # for paraview.
pvd_file << u
a1a2tu_out.close()


pie = assemble((1/(100))*( (1-uf(fR,fI)**2)**2/2 + uf(fR,fI).dx(0)*uf(fR,fI).dx(0) + uf(fR,fI).dx(0)*uf(fR,fI).dx(1) + uf(fR,fI).dx(1)*uf(fR,fI).dx(1) \
     + ( (a1-theta(fR,fI).dx(0))**2 + (a1-theta(fR,fI).dx(0))*(a2-theta(fR,fI).dx(1)) + (a2-theta(fR,fI).dx(1))**2 )*uf(fR,fI)**2 \
     + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )

print("================output of code========================")
print("gamma = ", gamma)
print("NN = ", NN)
print("H = ", float(H))
print("Energy density = ", pie)
print("tol = ", float(tol_test))

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

