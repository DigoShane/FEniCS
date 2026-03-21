#Here we solve the 1D Ginzbug Landau problem without an applied magnetic field and circular magnetic inhomogeneity.
#The problem is assumed 2D.
#Here we want to use Energy minimization method. We start off with Gradient Descent.
#HEre a is \ve{A}\cdot e_2 and m is the vertical components of the magnetization in the material.
#---------------------------------------------------------------------------------------------------------------
# The energy functional in non-dimensional form is presented below. For the derivation see "Overleaf.Superconductivity-Pradeep+Liping/Z7-MAgnet+Vortex.tex
# "/Section.Non-dimensopnal Complex valued Minimization problem".
# \int_{\Omega_s} (1-|u|^2)^2/2 + (\nabla u/\kappa)^2 + A^2u^2 + aex/2(\nabla m)^2 + \phi(m) + |\curl A - m|^2 dx
#where u is the magnitude of the complex valued normalized SC order parameter, m is the vertical magnetization and theta has been taken to 0. 
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#   b. The relaxation parameter
#   c. The absolute tolerance
#   d. Radius of the inhomogeneity.
#   e. Material parameters. Superconductor and inhomogeneity.
#2. When reading from and writing into respective files,
#   we are writing the lagrange multiplier as a constant function
#   When reading the functions, we interpolate onto a space VAu.
#======================================================================================================
#Things to keep in mind about writing this code:-
#1. Hasn't converged yet.
#======================================================================================================
#ISSUES WITH THE CODE:-

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


#Create mesh and define function space
lx = float(20)
kappa = Constant(2.0)
mesh = IntervalMesh(np.ceil(lx*10/kappa), 0, lx)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile
tol_prev = 1 # Initializing tol_prev to 1.


# Define functions
a = Function(V)
u = Function(V)
m = Function(V)
a_up = Function(V)
u_up = Function(V)
m_up = Function(V)

# Parameters
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = float(0); #float(input("External Magnetic field? -->"));
tol = float(0.000001) #float(input("absolute tolerance? --> "))
aex = float(1); #float(input("Exchange parameter? --> "));
m0 = float(input("What is the magnetization? --> "));
phi0 = float(input("Relative energy magnitude of the ferromagnetic state? --> "));
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
skp = int(input("Skip tolerance increase check? 1 for Yes, 0 for No --> "))

#To denote increase of tolerance
c = int(0)

#Defining the energy
Pi = ( (1-u**2)**2/2 + (u.dx(0)/kappa)**2 + (a*u)**2 + aex/2*(m.dx(0)**2) \
        + phi0*(m**2-m0**2)**2 + (a.dx(0)-m)**2 )*dx


#Defining the gradient
Fa = derivative(Pi, a)
Fu = derivative(Pi, u)
Fm = derivative(Pi, m)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 ##SC state
 #print("Using bulk SC as initial condition")
 A = interpolate( Expression("1+sin(2*x[0])", degree=2), V)
 U = interpolate( Expression("sin(x[0])", degree=2), V)
 M = interpolate( Expression("cos(x[0])", degree=2), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A = Function(V)
 U = Function(V)
 M = Function(V)
 a_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-0.xdmf")
 a_in.read_checkpoint(A,"a",0)
 u_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-1.xdmf")
 u_in.read_checkpoint(U,"u",0)
 m_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-2.xdmf")
 m_in.read_checkpoint(M,"m",0)
else:
 import sys
 sys.exit("Not a valid input for read_in.")

a_up.vector()[:] = A.vector()[:]
u_up.vector()[:] = U.vector()[:]
m_up.vector()[:] = M.vector()[:]

#========================================================================================================================

file1 = open("output.txt", "w")
L = ["The list of energies and tolerance at each iteraton:\n"]
file1.writelines(L)
file1.writelines(["=======================================================================\n"])
file1.close()


for tt in range(NN):
 a.vector()[:] = a_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]
 m.vector()[:] = m_up.vector()[:]

 Fa_vec = assemble(Fa)
 Fu_vec = assemble(Fu)
 Fm_vec = assemble(Fm)

 a_up.vector()[:] = a.vector()[:] - gamma*Fa_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]
 m_up.vector()[:] = m.vector()[:] - gamma*Fm_vec[:]
 tol_test = np.linalg.norm(np.asarray(Fa_vec.get_local()) + np.asarray(Fu_vec.get_local())\
            +np.asarray(Fm_vec.get_local()))
 #Defining the energy
 pie1 = assemble((1/lx)*( (1-u**2)**2/2 + (u.dx(0)/kappa)**2 + (a*u)**2 + aex/2*(m.dx(0)**2) \
        + phi0*(m**2-m0**2)**2 + (a.dx(0)-m)**2 )*dx )
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

##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-0.xdmf')
a1a2tu_out.write_checkpoint(a, "a", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-0.pvd") # for paraview. 
pvd_file << a
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-1.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-1.pvd") # for paraview. 
pvd_file << u
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-2.xdmf')
a1a2tu_out.write_checkpoint(m, "m", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-2.pvd") # for paraview.
pvd_file << m
a1a2tu_out.close()

#Defining observables
h = Function(V)
u = Function(V)
h = project(a.dx(0))

pie = assemble((1/lx)*( (1-u**2)**2/2 + (u.dx(0)/kappa)**2 + (a*u)**2 + aex/2*(m.dx(0)**2) \
        + phi0*(m**2-m0**2)**2 + (a.dx(0)-m)**2 )*dx )

print("Energy density =", pie)
print("gamma =", gamma)
print("Number of iterations =", NN)
print("External Magnetic field =", H)
print("absolute tolerance =", tol)
print("m0 =", m0)
print("phi =", phi0)
print("Exchange parameter =", aex)
print("Read from file? 1 for Yes, 0 for No =", read_in)

c = plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
c = plot(a)
plt.title(r"$A(x)$",fontsize=26)
plt.show()
c = plot(h)
plt.title(r"$h(x)$",fontsize=26)
plt.show()
c = plot(m)
plt.title(r"$m(x)$",fontsize=26)
plt.show()
c = plot(h-m)
plt.title(r"$B(x)$",fontsize=26)
plt.show()

