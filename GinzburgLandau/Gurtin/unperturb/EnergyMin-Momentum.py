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
from ufl import tanh
import matplotlib.pyplot as plt

import sys
np.set_printoptions(threshold=sys.maxsize)

#Parameters
lx = float(input("lx? -->"))
ly = float(input("ly? -->"))
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = Constant(input("External Magnetic field? -->"));
tol = float(input("absolute tolerance? --> "))
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
K11 = 3.25
K12 = 0.5
K22 = 2.71
M11 = 5
M12 = 0.212982
M22 = 3.23
T11 = 3.26
T12 = 3.24
T21 = -0.82
T22 = 2
kappa = float(2.0)
cr = float(1.5*kappa)
Ref_No = int(0) # No. of times u refine this.
method_no = int(0) #0->Gradient Descent, 1->Random SGD, 2->Nesterov, 3->Momentum 
tau = float(0.8) # stochastic parameter
if tau < 0 or tau > 1:
  sys.exit("tau cannot be negative")
if method_no != 3:
 method_no = int(3)
 print("This code is just for momentum, please check the other for vanilla.")

#Create mesh and define function space
Nx = np.ceil(lx*10/kappa)
Ny = np.ceil(ly*10/kappa)
Nx = int(2*np.ceil(Nx/2))
Ny = int(2*np.ceil(Ny/2))
tol_prev = 1 # setting this to 1 randomly.
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), Nx, Ny) 

#To denote increase of tolerance
c = int(0)

x = SpatialCoordinate(mesh)
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2
V = FunctionSpace(mesh, "Lagrange", 1)#This is for ExtFile

mesh.init(0,1)

#========================================================================================================================
# Define functions
a1 = Function(V)
a2 = Function(V)
u = Function(V)
a1_up = Function(V)
a2_up = Function(V)
u_up = Function(V)
#u1_up = Function(V)
#Temp functions to store the frechet derivatives
temp_a1 = Function(V)
temp_a2 = Function(V)
temp_u = Function(V)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

#Defining the energy
Pi = ( (1-u**2)**2/2 + K11*u.dx(0)*u.dx(0) + 2*K12*u.dx(0)*u.dx(1) + K22*u.dx(1)*u.dx(1) \
     + 2*T11*u.dx(0)*a1*u + 2*T12*u.dx(1)*a1*u + 2*T21*u.dx(0)*a2*u + 2*T22*u.dx(1)*a2*u \
     + ( M11*a1**2 + 2*M12*a1*a2 + M22*a2**2 )*u**2 + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

#Defining the gradients for each branch of the Riemann manifold.
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
Fu = derivative(Pi, u)
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
 A1 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? -x[1] : \
                             -exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *x[1]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=1), V)
 A2 = interpolate( Expression('sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? x[0] : \
                             exp(-sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *x[0]/sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=1), V)
 U = interpolate( Expression('sqrt(1-tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly)))*tanh(sqrt((x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) )', lx=lx, ly=ly, degree=1), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 U = Function(V)
 a1_in = XDMFFile("Gurtin-SCIsland-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in = XDMFFile("Gurtin-SCIsland-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 u_in = XDMFFile("Gurtin-SCIsland-2.xdmf")
 u_in.read_checkpoint(U,"u",0)
else:
 sys.exit("Not a valid input for read_in.")
#========================================================================================================================

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
u_up.vector()[:] = U.vector()[:]

#========================================================================================================================
## Determining the nodes to change
xcoord = mesh.coordinates()
v2d = vertex_to_dof_map(V)
d2v = dof_to_vertex_map(V)
disc_node = []
n = V.dim()                                                                      
d = mesh.geometry().dim()                                                        
dof_coordinates = V.tabulate_dof_coordinates()

##Marking the nodes near the discontinuity.
print("Marking the nodes near the discontinuity.")
for j,yy in enumerate(dof_coordinates):
 if yy[0] > 0.5*lx : # nodes to the right of the core.
  disc_node.append(j)

#========================================================================================================================
for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]

 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 Fu_vec = assemble(Fu)

 if tt == 0:
  ma1_vec = [0.0 for _ in range(len(Fa1_vec))]
  ma2_vec = [0.0 for _ in range(len(Fa2_vec))]
  mu_vec = [0.0 for _ in range(len(Fu_vec))]

 ma1_vec = [ii * tau for ii in ma1_vec]
 ma1_vec[:] = ma1_vec[:] - gamma*Fa1_vec[:]
 a1_up.vector()[:] = a1.vector()[:] + ma1_vec
 ma2_vec = [ii * tau for ii in ma2_vec]
 ma2_vec[:] = ma2_vec[:] - gamma*Fa2_vec[:]
 a2_up.vector()[:] = a2.vector()[:] + ma2_vec
 mu_vec = [ii * tau for ii in mu_vec]
 mu_vec[:] = mu_vec[:] - gamma*Fu_vec[:]
 u_up.vector()[:] = u.vector()[:] + mu_vec 

 #print(Fa1_vec.get_local()) # prints the vector.
 #print(np.linalg.norm(np.asarray(Fa1_vec.get_local()))) # prints the vector's norm.
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fu_vec.get_local()))
 #print(tol_test)
 if tt == 0 :
  tol_prev = tol_test
 else :
  if tol_test >= tol_prev:
   c=1
  else :
   tol_prev = tol_test

 #print(tol_test)
 if float(tol_test)  < tol :
  break

if c == 1:
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
a1a2tu_out = XDMFFile('Gurtin-SCIsland-2.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-2.pvd") # for paraview.
pvd_file << u
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-SCIsland-h.xdmf')
a1a2tu_out.write_checkpoint(h, "h", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-SCIsland-h.pvd") # for paraview.
pvd_file << h
a1a2tu_out.close()

pie = assemble((1/(lx*ly))*( (1-u**2)**2/2 + K11*u.dx(0)*u.dx(0) + 2*K12*u.dx(0)*u.dx(1) + K22*u.dx(1)*u.dx(1) \
     + 2*T11*u.dx(0)*a1*u + 2*T12*u.dx(1)*a1*u + 2*T21*u.dx(0)*a2*u + 2*T22*u.dx(1)*a2*u \
     + ( M11*a1**2 + 2*M12*a1*a2 + M22*a2**2 )*u**2 + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )

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
print("cr = ", cr)
print("Ref_No = ", Ref_No)
print("method_no = ", method_no)
print("tau = ", tau)
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

