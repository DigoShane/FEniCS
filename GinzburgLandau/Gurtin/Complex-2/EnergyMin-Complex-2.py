#Setup for this code is in OneNote.UH/Coding/Complex gurtin-2
#======================================================================================================
##Things to do:-
# Check bottom
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

#To move files
import shutil

#Parameters
lx = float(input("lx? -->"))
ly = float(input("ly? -->"))
gamma = float(input('Learning rate? -->')) # Learning rate.
#gamma = float(0.01) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
Hi = float(input("External Magnetic field? -->"));
H = Constant(Hi);
tol = float(0.000001) #float(input("absolute tolerance? --> "))
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
K11 = 4
K12 = -2
K22 = 3
T11 = 2.5
T12 = -1.5
T21 = 2
T22 = -1.2
kappa = float(2.0)

#Defining the matrices
K = as_matrix([[K11,K12],[K12,K22]])
T = as_matrix([[T11,T12],[T21,T22]])

#Create mesh and define function space
Nx = np.ceil(lx*100/kappa)
Ny = np.ceil(ly*100/kappa)
Nx = int(2*np.ceil(Nx/2))
Ny = int(2*np.ceil(Ny/2))
tol_prev = 1 # setting this to 1 randomly.
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), Nx, Ny) 

#To denote increase of tolerance
c = int(0)

x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 1)#This is for ExtFile
Va = VectorFunctionSpace(mesh, "Lagrange", 1, 2)
Ae = interpolate( Expression(("0","H*x[0]"), H=H, degree=2), Va) #vec potential for external field.

#========================================================================================================================
# Define functions
a = Function(Va)
fR = Function(V)
fI = Function(V)
a_up = Function(Va)
fR_up = Function(V)
fI_up = Function(V)

Pi = ( (1-(fR**2+fI**2)**2)**2/2 + 4*fR**2*inner(grad(fR) , K*grad(fR)) + 8*fR*fI*inner(grad(fR) , K*grad(fI)) + 4*fI**2*inner(grad(fI) , K*grad(fI))\
     + (fR**2+fI**2)**2*dot(a,a) + inner(fI*grad(fR) - fR*grad(fI), fI*grad(fR) - fR*grad(fI)) + 2*(fR**2+fI**2)*inner(a,fI*grad(fR) - fR*grad(fI)) \
     + 4*(fR**2+fI**2)*inner(a,T*(fR*grad(fR)+fI*grad(fI))) + 4*inner(fI*grad(fR)-fR*grad(fI),T*(fR*grad(fR)+fI*grad(fI))) + curl(a-Ae)**2 )*dx

#Defining the gradients for each branch of the Riemann manifold.
Fa = derivative(Pi, a)
FfR = derivative(Pi, fR)
FfI = derivative(Pi, fI)


#========================================================================================================================
##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 ##SC state
 print("Using bulk SC as initial condition")
 A = interpolate( Expression(("0","0.5"), degree=2), Va)#SC phase as initial cond.
 FR = interpolate( Expression('0.7', degree=1), V)
 FI = interpolate( Expression('0.7', degree=1), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A = Function(Va)
 FR = Function(V)
 FI = Function(V)
 a_in = XDMFFile("Gurtin-Complex-0.xdmf")
 a_in.read_checkpoint(A,"a",0)
 fR_in = XDMFFile("Gurtin-Complex-1.xdmf")
 fR_in.read_checkpoint(FR,"fR",0)
 fI_in = XDMFFile("Gurtin-Complex-2.xdmf")
 fI_in.read_checkpoint(FI,"fI",0)
 shutil.move("Gurtin-Complex-0.xdmf", "Older/Gurtin-Complex-0.xdmf") 
 shutil.move("Gurtin-Complex-1.xdmf", "Older/Gurtin-Complex-1.xdmf") 
 shutil.move("Gurtin-Complex-2.xdmf", "Older/Gurtin-Complex-2.xdmf") 
else:
 sys.exit("Not a valid input for read_in.")
#========================================================================================================================

a_up.vector()[:] = A.vector()[:]
fR_up.vector()[:] = FR.vector()[:]
fI_up.vector()[:] = FI.vector()[:]

#========================================================================================================================
print("time taken so far",time.time()-time0)
for tt in range(NN):
 #time1b4 = time.time()
 a.vector()[:] = a_up.vector()[:]
 fR.vector()[:] = fR_up.vector()[:]
 fI.vector()[:] = fI_up.vector()[:]

 Fa_vec = assemble(Fa)
 FfR_vec = assemble(FfR)
 FfI_vec = assemble(FfI)

 a_up.vector()[:] = a.vector()[:] - gamma*Fa_vec[:]
 fR_up.vector()[:] = fR.vector()[:] - gamma*FfR_vec[:]
 fI_up.vector()[:] = fI.vector()[:] - gamma*FfI_vec[:]

 tol_test = np.linalg.norm(np.asarray(Fa_vec.get_local())) + np.linalg.norm(np.asarray(FfR_vec.get_local()) + np.asarray(FfI_vec.get_local()))
            
 #if (tt%10)==0:
 # print(tol_test)
 # print("time taken so far",time.time()-time1b4)

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
h = project(curl(a))
u = Function(V)
u = project(fR**2+fI**2)


##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('Gurtin-Complex-0.xdmf')
a1a2tu_out.write_checkpoint(a, "a", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-Complex-0.pvd") # for paraview. 
pvd_file << a
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-Complex-1.xdmf')
a1a2tu_out.write_checkpoint(fR, "fR", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-Complex-1.pvd") # for paraview.
pvd_file << fR
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-Complex-2.xdmf')
a1a2tu_out.write_checkpoint(fI, "fI", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-Complex-2.pvd") # for paraview.
pvd_file << fI
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-Complex-h.xdmf')
a1a2tu_out.write_checkpoint(h, "h", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-Complex-h.pvd") # for paraview.
pvd_file << h
a1a2tu_out.close()
a1a2tu_out = XDMFFile('Gurtin-Complex-u.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("Gurtin-Complex-u.pvd") # for paraview.
pvd_file << u
a1a2tu_out.close()

pie = assemble((1/(lx*ly))*( (1-(fR**2+fI**2)**2)**2/2 + 4*fR**2*inner(grad(fR) , K*grad(fR)) + 8*fR*fI*inner(grad(fR) , K*grad(fI)) + 4*fI**2*inner(grad(fI) , K*grad(fI))\
     + (fR**2+fI**2)**2*dot(a,a) + inner(fI*grad(fR) - fR*grad(fI), fI*grad(fR) - fR*grad(fI)) + 2*(fR**2+fI**2)*inner(a,fI*grad(fR) - fR*grad(fI))\
     + 4*(fR**2+fI**2)*inner(a,T*(fR*grad(fR)+fI*grad(fI))) + 4*inner(fI*grad(fR)-fR*grad(fI),T*(fR*grad(fR)+fI*grad(fI))) + curl(a-Ae)**2 )*dx )


print("================output of code========================")
print("gamma = ", gamma)
print("kappa = ", kappa)
print("lx = ", lx)
print("ly = ", ly)
print("Nx = ", Nx)
print("Ny = ", Ny)
print("NN = ", NN)
print("H = ", Hi)
print("Energy density = ", pie)
print("tol = ", float(tol_test))
print("======================================================")
print("Things to do:-")
print("1. Set T_m=0, and M_m=0 and remove the demag term. (IGNORED)")
print("2. Create a folder called Older where you can save the previous input files.")
print("   When increasing gamma goes wrong, you can get back the older starting point. (DONE)")
print("3. Find out what is taking too long. (DONE) ")
print(" 3a. Set timer over loop. (done)")
print(" 3b. Check how time changes as you ggincrease the no. of iterations. (done)")
print("     --> Doesnt change with iteration no. Moreover printing the tolerance+time taken takes the same time as printing just the time taken. Therefore this is not the main source of delay.")
print(" 3c. Expand out the energy expression. Check if timer reports a decrease. ")
print("     ---> This worked the best reduced it to half the original time taken. (done) ")
print("4. Add all the terms. Repeating the above experiment again. (DONE) ")
print(" 4a. Adding the demag term. (done)")
print(" 4b. Adding the other terms. (done).")
print("5. Make a vector valued function to save computation time. (DONE) ") 
print("   This really helped a lot. It reduce the compute time by a lot. ") 
print("6. Make sure that the use of the terms is correct. (PERFORMING) ") 
print(" 6a. Maybe test with simple solution of an isotropic problem where you know the solution.")
print("7. MAnipulate the parameters to show different solutions.")

c = plot(u)
plt.title(r"$u(x)$ -- Complex Gurtin",fontsize=26)
cb = plt.colorbar(c)
plt.show()
c = plot(h)
plt.title(r"$h(x)$",fontsize=26)
cb = plt.colorbar(c)
plt.show()

time1 = time.time()

print(str(datetime.timedelta(seconds=time1-time0)), "sec for code to run")
print("time = ", time1-time0)

