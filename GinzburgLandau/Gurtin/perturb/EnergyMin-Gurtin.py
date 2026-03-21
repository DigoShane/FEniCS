#Here we solve the 2D Gurtin modification problem with an applied magnetic field.
#We are interested in the perturbed solution.
#------------------------------------------------------------------------------------------------------
# For details on progress, visit UH OneNote. superConductivity/Coding/ General Gurtin'S tensor  
#
#======================================================================================================
#HEre fR=Real(f) and f_I = Imag(f)
# with f is the normalized complex valued wave function.
# The domain is set up as [-lx/2,lx/2]x[-ly/2,ly/2]
#======================================================================================================
#THINGS TO BE CAREFUL OF:-
#1. Gauge might need fixing
#2. option to stop and restart if tolerance increases.
#======================================================================================================
#ISSUES WITH THE CODE:-

import time # timing for performance test.
t0 = time.time()

import dolfin
print(f"DOLFIN version: {dolfin.__version__}")
from dolfin import *
import fenics as fe
import numpy as np
import ufl
print(f" UFL version: {ufl.__version__}")
from ufl import tanh
import matplotlib.pyplot as plt
#import mshr

import sys
np.set_printoptions(threshold=sys.maxsize)

#Parameters
print("================input to code========================")
lx = float(input("lx? --> "))
ly = float(input("ly? --> "))
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = Constant(input("External Magnetic field? -->"));
tol = float(input("absolute tolerance? --> "))
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
K1 = 0.1
K2 = 1
M1 = 1
M2 = 1
T1 = 0.2
T2 = 0.5


#Create mesh and define function space
N1 = int(1+np.ceil(lx*10/K1))
N2 = int(1+np.ceil(ly*10/K2))
mesh = RectangleMesh(Point(-lx*0.5, -ly*0.5), Point(lx*0.5, ly*0.5), N1, N2) 
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 2)

# Define functions
#unperturbed variables
a1 = Function(V)
a2 = Function(V)
fR = Function(V)
fI = Function(V)
a1_up = Function(V)
a2_up = Function(V)
fR_up = Function(V)
fI_up = Function(V)
#Temp functions to store the frechet derivatives
temp_a1 = Function(V)
temp_a2 = Function(V)
temp_fR = Function(V)
temp_fI = Function(V)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

def fsq(fR,fI):
    return fR*fR+fI*fI

def PE(fR,fI):
    return (1-fsq(fR,fI))*(1-fsq(fR,fI))/2

def KE(fR,fI):
    return (K1*fsq(fR,fI).dx(0)*fsq(fR,fI).dx(0) + K2*fsq(fR,fI).dx(1)*fsq(fR,fI).dx(1))/(4*fsq(fR,fI))

def Demag(a1,a2):
    return (curl(a1,a2) - H)*(curl(a1,a2) - H)

def gradtheta1(fR,fI):
    return  (fI*fR.dx(0)-fR*fI.dx(0))/fsq(fR,fI)

def gradtheta2(fR,fI):
    return  (fI*fR.dx(1)-fR*fI.dx(1))/fsq(fR,fI)

def Gurtin(a1,a2,fR,fI):
    return T1*fsq(fR,fI).dx(0)*(a1+gradtheta1(fR,fI)) + T2*fsq(fR,fI).dx(1)*(a2+gradtheta2(fR,fI))

def MA0u(a1,a2,fR,fI):
    return M1*(a1+gradtheta1(fR,fI))*(a1+gradtheta1(fR,fI))*fsq(fR,fI) + M2*(a2+gradtheta2(fR,fI))*(a2+gradtheta2(fR,fI))*fsq(fR,fI)

#======================================================================================================================
#Solving the problem for the unperturbed problem
#======================================================================================================================

#Defining the unperturbed energy
#Pi = ( PE(fR,fI) + KE(fR,fI) + Demag(a1,a2) + Gurtin(a1,a2,fR,fI) + MA0u(a1,a2,fR,fI) )*dx

Pi = ( (1-fR*fR-fI*fI)*(1-fR*fR-fI*fI)/2\
        + (fR*fR*fR.dx(0)*K1*fR.dx(0) + fR*fR*fR.dx(1)*K2*fR.dx(1))/(fR*fR+fI*fI)\
       +(2*fR*fI*fR.dx(0)*K1*fI.dx(0)+2*fR*fI*fR.dx(1)*K2*fI.dx(1))/(fR*fR+fI*fI)\
        + (fI*fI*fI.dx(0)*K1*fI.dx(0) + fI*fI*fI.dx(1)*K2*fI.dx(1))/(fR*fR+fI*fI)\
        + (fR*fR.dx(0)+fI*fI.dx(0))*2*T1*(a1 +(fI*fR.dx(0)-fR*fI.dx(0))/(fR*fR+fI*fI)) \
        + (fR*fR.dx(1)+fI*fI.dx(1))*2*T2*(a2 +(fI*fR.dx(1)-fR*fI.dx(1))/(fR*fR+fI*fI)) \
        + (fR*fR+fI*fI)*(a1 +(fI*fR.dx(0)-fR*fI.dx(0))/(fR*fR+fI*fI))*M1*(a1 +(fI*fR.dx(0)-fR*fI.dx(0))/(fR*fR+fI*fI)) \
        + (fR*fR+fI*fI)*(a2 +(fI*fR.dx(1)-fR*fI.dx(1))/(fR*fR+fI*fI))*M2*(a2 +(fI*fR.dx(1)-fR*fI.dx(1))/(fR*fR+fI*fI)) \
        + (a2.dx(0)-a1.dx(1) -H)*(a2.dx(0)-a1.dx(1) -H) )*dx

#Defining the gradient
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
FfR = derivative(Pi, fR)
FfI = derivative(Pi, fI)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 ##SC state
 #print("Using bulk SC as initial condition")
 #A1 = interpolate( Expression("0.0", degree=2), V)
 #A2 = interpolate( Expression("0.0", degree=2), V)
 #T = interpolate( Expression("1.0", degree=2), V)
 #U = interpolate( Expression("1.0", degree=2), V)
 ##Modified normal state
 #print("Using modified bulk Normal as initial condition")
 #A1 = interpolate( Expression("0.0", degree=2), V)
 #A2 = interpolate( Expression("H*x[0]", H=H, degree=2), V)
 #T = interpolate( Expression("x[1]", degree=2), V)
 #U = interpolate( Expression("x[0]", degree=2), V)
 ##Vortex Solution.
 print("Using Vortex solution")
 A1 = interpolate( Expression('sqrt(x[0]*x[0]+x[1]*x[1]) <= r + DOLFIN_EPS ? -x[1] : \
                             -exp(-sqrt(x[0]*x[0]+x[1]*x[1]))*x[1]/sqrt(x[0]*x[0]+x[1]*x[1])*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=1, degree=2), V)
 A2 = interpolate( Expression('sqrt(x[0]*x[0]+x[1]*x[1]) <= r + DOLFIN_EPS ? x[0] : \
                             exp(-sqrt(x[0]*x[0]+x[1]*x[1]))*x[0]/sqrt(x[0]*x[0]+x[1]*x[1])*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=1, degree=2), V)
 FR = interpolate( Expression('sqrt(x[0]*x[0]+x[1]*x[1]) <= r + DOLFIN_EPS ? x[0] : \
                             tanh(K*sqrt(x[0]*x[0]+x[1]*x[1]))*x[0]/sqrt(x[0]*x[0]+x[1]*x[1])', \
                                lx=lx, ly=ly, r=0.3517, K=1, degree=2), V)
 FI = interpolate( Expression('sqrt(x[0]*x[0]+x[1]*x[1]) <= r + DOLFIN_EPS ? x[1] : \
                             tanh(K*sqrt(x[0]*x[0]+x[1]*x[1]))*x[1]/sqrt(x[0]*x[0]+x[1]*x[1])', \
                                lx=lx, ly=ly, r=0.3517, K=1, degree=2), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 FR = Function(V)
 FI = Function(V)
 a1_in =  XDMFFile("Gurtin-2DEnrg-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in =  XDMFFile("Gurtin-2DEnrg-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 fR_in =  XDMFFile("Gurtin-2DEnrg-2.xdmf")
 fR_in.read_checkpoint(FR,"fR",0)
 fI_in =  XDMFFile("Gurtin-2DEnrg-3.xdmf")
 fI_in.read_checkpoint(FI,"fI",0)
 #plot(u)
 #plt.title(r"$u(x)-b4$",fontsize=26)
 #plt.show()
else:
 import sys
 sys.exit("Not a valid input for read_in.")

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
fR_up.vector()[:] = FR.vector()[:]
fI_up.vector()[:] = FI.vector()[:]

c = plot(FR) #plot(fR_up)
plt.title(r"$fR(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(FI) #plot(fI_up)
plt.title(r"$fI(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(A1) #plot(a1)
plt.title(r"$a1(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(A2) #plot(a2)
plt.title(r"$a2(x)$",fontsize=26)
plt.colorbar(c)
plt.show()

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
 temp_a1.vector()[:] = Fa1_vec[:]
 temp_a2.vector()[:] = Fa2_vec[:]
 temp_fR.vector()[:] = FfR_vec[:]
 temp_fI.vector()[:] = FfI_vec[:]
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(FfR_vec.get_local()))\
           +np.linalg.norm(np.asarray(FfI_vec.get_local()))
 print(tol_test)
 if float(tol_test)  < tol :
  break

u = Function(V)
h = Function(V)
a01 = Function(V)
a02 = Function(V)
u = project(sqrt(fI**2+fR**2))
h = project(curl(a1,a2))
a01 = project(a1 + (fI*fR.dx(0)-fR*fI.dx(0))/u**2)
a02 = project(a2 + (fI*fR.dx(1)-fR*fI.dx(1))/u**2)

##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('Gurtin-2DEnrg-0.xdmf')
a1a2tu_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-2DEnrg-0.pvd") # for paraview. 
pvd_file << a1
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-2DEnrg-1.xdmf')
a1a2tu_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-2DEnrg-1.pvd") # for paraview. 
pvd_file << a2
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-2DEnrg-2.xdmf')
a1a2tu_out.write_checkpoint(fR, "fR", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-2DEnrg-2.pvd") # for paraview.
pvd_file << fR
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-2DEnrg-3.xdmf')
a1a2tu_out.write_checkpoint(fI, "fI", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-2DEnrg-3.pvd") 
pvd_file << fI
a1a2tu_out.close()

a1a2tu_out = XDMFFile('Gurtin-2DEnrg-u.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-2DEnrg-u.pvd") 
pvd_file << u
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-2DEnrg-h.xdmf')
a1a2tu_out.write_checkpoint(h, "h", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-2DEnrg-h.pvd") 
pvd_file << h
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-2DEnrg-a01.xdmf')
a1a2tu_out.write_checkpoint(a01, "a01", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-2DEnrg-a01.pvd") 
pvd_file << a01
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-2DEnrg-a02.xdmf')
a1a2tu_out.write_checkpoint(a01, "a02", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-2DEnrg-a02.pvd") 
pvd_file << a02
a1a2tu_out.close()

pie = assemble( (1/(lx*ly))*( PE(fR,fI) + KE(fR,fI) + Demag(a1,a2) + Gurtin(a1,a2,fR,fI) + MA0u(a1,a2,fR,fI) )*dx )


print("================output of code========================")
print("Energy density is", pie)
print("gamma = ", gamma)
print("lx = ", lx)
print("ly = ", ly)
print("NN = ", NN)
print("H = ", float(H))
print("tol = ", tol, ", ", float(tol_test))
print("read_in = ", read_in)


c = plot(a1)
plt.title(r"$A_1(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(a2)
plt.title(r"$A_2(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(fR)
plt.title(r"$Re(f(x))$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(fI)
plt.title(r"$Im(f(x))$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(u)
plt.title(r"$|u(x)|$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(h)
plt.title(r"$h(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(a01)
plt.title(r"$a01(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(a02)
plt.title(r"$a02(x)$",fontsize=26)
plt.colorbar(c)
plt.show()



t1 = time.time()

print("time taken for code to run = ", t1-t0)
