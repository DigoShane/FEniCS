#Here we solve the 2D Ginzbug Landau problem with an applied magnetic field.
#We are interested in the perturbed solution.
#------------------------------------------------------------------------------------------------------
# For details on progress, visit UH OneNote. superConductivity/Coding/ perturbative gurtin tensor - theory
#Check p13 onwards.
#======================================================================================================
#HEre uR=Real(\psi) and u_I = Imag(\psi)
# with \psi is the normalized complex valued wave function.
# The domain is set up as [-lx/2,lx/2]x[-ly/2,ly/2]
#======================================================================================================
#THINGS TO BE CAREFUL OF:-
#1. Might need to fix the gauge to find the solution.
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
kappa = Constant(2.0)
lx = float(input("lx? --> "))
ly = float(input("ly? --> "))
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = Constant(input("External Magnetic field? -->"));
tol = float(input("absolute tolerance? --> "))
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
F1 = 1
F2 = 2
Run = int(input("Run perturbed problem? 1 for Yes, 0 for No --> "))
NN_p = int(1000000)
gamma_p = float(0.1)


#Create mesh and define function space
mesh = RectangleMesh(Point(-lx*0.5, -ly*0.5), Point(lx*0.5, ly*0.5), 1+np.ceil(lx*10/kappa), 1+np.ceil(ly*10/kappa), "crossed") # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"
#mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), 3, 3) # "crossed means that first it will partition the ddomain into Nx x Ny rectangles. Then each rectangle is divided into 4 triangles forming a cross"
x = SpatialCoordinate(mesh)
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2
V = FunctionSpace(mesh, "Lagrange", 2)

# Define functions
#unperturbed variables
a1 = Function(V)
a2 = Function(V)
uR = Function(V)
uI = Function(V)
a1_up = Function(V)
a2_up = Function(V)
uR_up = Function(V)
uI_up = Function(V)
#perturbed variables
b1 = Function(V)
b2 = Function(V)
v  = Function(V)
b1_up = Function(V)
b2_up = Function(V)
v_up = Function(V)
#Temp functions to store the frechet derivatives
temp_a1 = Function(V)
temp_a2 = Function(V)
temp_uR = Function(V)
temp_uI = Function(V)
temp_b1 = Function(V)
temp_b2 = Function(V)
temp_v  = Function(V)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)


#======================================================================================================================
#Solving the problem for the unperturbed problem
#======================================================================================================================

#Defining the unperturbed energy
Pi = ( (1-uR*uR-uI*uI)*(1-uR*uR-uI*uI)/2 + (1/kappa**2)*inner(grad(uR), grad(uR)) + (1/kappa**2)*inner(grad(uI), grad(uI)) + (a1*a1+a2*a2)*(uR*uR+uI*uI)\
     + (2/kappa)*a1*(uI*uR.dx(0)-uR*uI.dx(0)) + (2/kappa)*a2*(uI*uR.dx(1)-uR*uI.dx(1))  + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

#Defining the gradient
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
FuR = derivative(Pi, uR)
FuI = derivative(Pi, uI)


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
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=2), V)
 A2 = interpolate( Expression('sqrt(x[0]*x[0]+x[1]*x[1]) <= r + DOLFIN_EPS ? x[0] : \
                             exp(-sqrt(x[0]*x[0]+x[1]*x[1]))*x[0]/sqrt(x[0]*x[0]+x[1]*x[1])*1/K', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=2), V)
 UR = interpolate( Expression('sqrt(x[0]*x[0]+x[1]*x[1]) <= r + DOLFIN_EPS ? x[0] : \
                             tanh(K*sqrt(x[0]*x[0]+x[1]*x[1]))*x[0]/sqrt(x[0]*x[0]+x[1]*x[1])', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=2), V)
 UI = interpolate( Expression('sqrt(x[0]*x[0]+x[1]*x[1]) <= r + DOLFIN_EPS ? x[1] : \
                             tanh(K*sqrt(x[0]*x[0]+x[1]*x[1]))*x[1]/sqrt(x[0]*x[0]+x[1]*x[1])', \
                                lx=lx, ly=ly, r=0.3517, K=kappa, degree=2), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 UR = Function(V)
 UI = Function(V)
 a1_in =  XDMFFile("GL-2DEnrg-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in =  XDMFFile("GL-2DEnrg-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 uR_in =  XDMFFile("GL-2DEnrg-2.xdmf")
 uR_in.read_checkpoint(UR,"uR",0)
 uI_in =  XDMFFile("GL-2DEnrg-3.xdmf")
 uI_in.read_checkpoint(UI,"uI",0)
 #plot(u)
 #plt.title(r"$u(x)-b4$",fontsize=26)
 #plt.show()
else:
 import sys
 sys.exit("Not a valid input for read_in.")

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
uR_up.vector()[:] = UR.vector()[:]
uR_up.vector()[:] = UR.vector()[:]
uI_up.vector()[:] = UI.vector()[:]

c = plot(UR) #plot(uR_up)
plt.title(r"$uR(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(UI) #plot(uI_up)
plt.title(r"$uI(x)$",fontsize=26)
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
 uR.vector()[:] = uR_up.vector()[:] 
 uI.vector()[:] = uI_up.vector()[:] 
 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 FuR_vec = assemble(FuR)
 FuI_vec = assemble(FuI)
 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 uR_up.vector()[:] = uR.vector()[:] - gamma*FuR_vec[:]
 uI_up.vector()[:] = uI.vector()[:] - gamma*FuI_vec[:]
 temp_a1.vector()[:] = Fa1_vec[:]
 temp_a2.vector()[:] = Fa2_vec[:]
 temp_uR.vector()[:] = FuR_vec[:]
 temp_uI.vector()[:] = FuI_vec[:]
 #c = plot(temp_uR)
 #plt.title(r"$F_{uR}(x)$",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #c = plot(temp_uI)
 #plt.title(r"$F_{uI}(x)$",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #c = plot(temp_a1)
 #plt.title(r"$F_{a1}(x)$",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #c = plot(temp_a2)
 #plt.title(r"$F_{a2}(x)$",fontsize=26)
 #plt.colorbar(c)
 #plt.show()
 #print(np.linalg.norm(np.asarray(Fa1_vec.get_local()))) # prints the vector's norm.
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(FuR_vec.get_local()))\
           +np.linalg.norm(np.asarray(FuI_vec.get_local()))
 print(tol_test)
 if float(tol_test)  < tol :
  break

u = Function(V)
h = Function(V)
a01 = Function(V)
a02 = Function(V)
u = project(sqrt(uI**2+uR**2))
h = project(curl(a1,a2))
a01 = project(a1 + (uI*uR.dx(0)-uR*uI.dx(0))/kappa/u**2)
a02 = project(a2 + (uI*uR.dx(1)-uR*uI.dx(1))/kappa/u**2)

##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('GL-2DEnrg-0.xdmf')
a1a2tu_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-0.pvd") # for paraview. 
pvd_file << a1
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-1.xdmf')
a1a2tu_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-1.pvd") # for paraview. 
pvd_file << a2
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-2.xdmf')
a1a2tu_out.write_checkpoint(uR, "uR", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-2.pvd") # for paraview.
pvd_file << uR
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-3.xdmf')
a1a2tu_out.write_checkpoint(uI, "uI", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-3.pvd") 
pvd_file << uI
a1a2tu_out.close()

a1a2tu_out = XDMFFile('GL-2DEnrg-u.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-u.pvd") 
pvd_file << u
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-h.xdmf')
a1a2tu_out.write_checkpoint(h, "h", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-h.pvd") 
pvd_file << h
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-a01.xdmf')
a1a2tu_out.write_checkpoint(a01, "a01", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-a01.pvd") 
pvd_file << a01
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-a02.xdmf')
a1a2tu_out.write_checkpoint(a01, "a02", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-a02.pvd") 
pvd_file << a02
a1a2tu_out.close()

pie = assemble( (1/(lx*ly))*( (1-uR**2-uI**2)**2/2 + (1/kappa**2)*inner(grad(uR), grad(uR)) + (1/kappa**2)*inner(grad(uI), grad(uI)) + (a1**2+a2**2)*(uR**2+uI**2)\
     + (2/kappa)*a1*(uI*uR.dx(0)-uR*uI.dx(0)) + (2/kappa)*a2*(uI*uR.dx(1)-uR*uI.dx(1))  + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )

print("================output of code========================")
print("Energy density is", pie)
print("gamma = ", gamma)
print("kappa = ", float(kappa))
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
c = plot(uR)
plt.title(r"$Re(u(x))$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(uI)
plt.title(r"$Im(u(x))$",fontsize=26)
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



#======================================================================================================================
#Solving the perturbed problem
#======================================================================================================================
if Run == 1:

 #Defining the perturbed energy
 dPi = ( (3*u**2-1)*v**2 + (1/kappa**2)*inner(grad(v), grad(v)) + 2*F1*(u*v.dx(0)+ v*u.dx(0))*a01 + 2*F2*(u*v.dx(1)+ v*u.dx(1)) \
       + 2*F1*u*u.dx(0)*b1 + 2*F2*u*u.dx(1)*b2 + a01**2*v**2 + a02**2*v**2 + b1**2*u**2 + b2**2*u**2 + 4*a01*b1*u*v + 4*a02*b2*u*v\
       + inner(curl(b1 ,b2), curl(b1 ,b2)) )*dx
 
 
 #Defining the gradient
 Fpb1 = derivative(dPi, b1)
 Fpb2 = derivative(dPi, b2)
 Fpv  = derivative(dPi, v)
 
 
 #Using 0 as the initial condition for the perturbed problem.
 print("Using 0 as the initial condition for the perturbed problem.")
 B1 = interpolate( Expression("2.0", degree=2), V)
 B2 = interpolate( Expression("3.0", degree=2), V)
 V = interpolate( Expression("1.0", degree=2), V)
 
 b1_up.vector()[:] = B1.vector()[:]
 b2_up.vector()[:] = B2.vector()[:]
 v_up.vector()[:]  = V.vector()[:]
 
 
 for tt in range(NN_p):
  b1.vector()[:] = b1_up.vector()[:]
  b2.vector()[:] = b2_up.vector()[:]
  v.vector()[:]  = v_up.vector()[:] 
  Fpb1_vec = assemble(Fpb1)
  Fpb2_vec = assemble(Fpb2)
  Fpv_vec  = assemble(Fpv)
  b1_up.vector()[:] = b1.vector()[:] - gamma_p*Fpb1_vec[:]
  b2_up.vector()[:] = b2.vector()[:] - gamma_p*Fpb2_vec[:]
  v_up.vector()[:]  = v.vector()[:] - gamma_p*Fpv_vec[:]
  tol_test_per = np.linalg.norm(np.asarray(Fpb1_vec.get_local()))\
            +np.linalg.norm(np.asarray(Fpb2_vec.get_local()))\
            +np.linalg.norm(np.asarray(Fpv_vec.get_local()))
  print(tol_test_per)
  if float(tol_test_per)  < float(tol*100) :
   break
 
 ##Save solution in a .xdmf file and for paraview.
 b1b2v_out = XDMFFile('GL-perturb-2DEnrg-0.xdmf')
 b1b2v_out.write_checkpoint(b1, "b1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
 pvd_file = File("GL-perturb-2DEnrg-0.pvd") # for paraview. 
 pvd_file << b1
 b1b2v_out.close()
 b1b2v_out = XDMFFile('GL-perturb-2DEnrg-1.xdmf')
 b1b2v_out.write_checkpoint(b2, "b2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
 pvd_file = File("GL-perturb-2DEnrg-1.pvd") # for paraview. 
 pvd_file << b2
 b1b2v_out.close()
 b1b2v_out = XDMFFile('GL-perturb-2DEnrg-2.xdmf')
 b1b2v_out.write_checkpoint(v, "v", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
 pvd_file = File("GL-perturb-2DEnrg-2.pvd") # for paraview.
 pvd_file << v
 b1b2v_out.close()
 
 
 c = plot(b1)
 plt.title(r"$B_1(x)$",fontsize=26)
 plt.colorbar(c)
 plt.show()
 c = plot(b2)
 plt.title(r"$B_2(x)$",fontsize=26)
 plt.colorbar(c)
 plt.show()
 c = plot(v)
 plt.title(r"$v(x)$",fontsize=26)
 plt.colorbar(c)
 plt.show()
 


t1 = time.time()

print("time taken for code to run = ", t1-t0)
