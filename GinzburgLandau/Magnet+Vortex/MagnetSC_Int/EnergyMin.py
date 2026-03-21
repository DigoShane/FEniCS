#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field and circular magnetic inhomogeneity.
#The problem is assumed 2D.
#Here we want to use Energy minimization method. We start off with Gradient Descent.
#HEre a is \ve{A}, u is u and m3 is the components of the magnetization of the inhomogeneity.
#---------------------------------------------------------------------------------------------------------------
# The energy functional in non-dimensional form is presented below. For the derivation see "Overleaf.Superconductivity-Pradeep+Liping/Z7-MAgnet+Vortex.tex
# "/Section.Non-dimensopnal Complex valued Minimization problem".
# \int_{x>0} (1-u^2)^2/2 + (\nabla u/\kappa)^2 + A^2u^2 dx +\int_{x<0} aex(\nabla m)^2/2 + \phi(m) - 2H.m dx + \int_{\Scr{R}} |\curl A - H - m\mathbbm{1}_{x<0}|^2 dx
#where u is the normalized SC order parameter. \Omega_S={x|x>0} is the superconducting domain and \Omega_m={x|x<0} is the magnetic inhomogeneity domain.
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#   b. The relaxation parameter
#   c. The absolute tolerance
#   e. Material parameters. Superconductor and inhomogeneity.
#======================================================================================================
#Things to keep in mind about writing this code:-
#1. Define a function to evaluate the curl
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
from ufl import tanh
import matplotlib.pyplot as plt
#import mshr
import sys
np.set_printoptions(threshold=sys.maxsize)


#Create mesh and define function space
lx = float(10)
kappa = Constant(2.0)
mesh = IntervalMesh(np.ceil(lx*10/kappa), -lx, lx)
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
H = float(input("External Magnetic field? -->"));
tol = float(0.000001) #float(input("absolute tolerance? --> "))
aex = float(input("Exchange parameter? --> "));
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))


#Defining the characteristic function. for the inhomogenity
k = Function(V)
k = interpolate( Expression("x[0] < 0 ? 1 : 0", degree=2), V)

#Defining the energy
Pi = ( (1-k)*(1-u**2)**2/2 + (1-k)*(u.dx(0)/kappa)**2 + (1-k)*(a*u)**2 \
      + k*aex/2*(m.dx(0)**2) + k*m**2 - k*2*H*m \
      + inner( a.dx(0) - H - k*m, a.dx(0) - H - k*m ) )*dx


#Defining the gradient
Fa = derivative(Pi, a)
Fu = derivative(Pi, u)
Fm = derivative(Pi, m)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 #SC state
 print("Using bulk SC as initial condition")
 A = interpolate( Expression("0.0", degree=2), V)
 U = interpolate( Expression("1.0", degree=2), V)
 M = interpolate( Expression("0.0", degree=2), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A = Function(V)
 U = Function(V)
 M = Function(V)
 a_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-0.xdmf")
 a_in.read_checkpoint(A,"a",0)
 U_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-1.xdmf")
 U_in.read_checkpoint(U,"u",0)
 m_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-2.xdmf")
 m_in.read_checkpoint(M,"m",0)
else:
 import sys
 sys.exit("Not a valid input for read_in.")

a_up.vector()[:] = A.vector()[:]
u_up.vector()[:] = U.vector()[:]
m_up.vector()[:] = M.vector()[:]

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
 tol_test = np.linalg.norm(np.asarray(Fa_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fu_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fm_vec.get_local()))
 print(tol_test)
 if tt == 0 :
  tol_prev = tol_test
 else :
  if tol_test >= tol_prev:
   print(tol_test)
   sys.exit("The tolerance just increased.")
  else :
   tol_prev = tol_test
 if float(tol_test)  < tol :
  break
 
#print(tol_test)

##Save solution in a .xdmf file and for paraview.
aum_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-0.xdmf')
aum_out.write_checkpoint(a, "a", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-0.pvd") # for paraview. 
pvd_file << a
aum_out.close()
aum_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-1.xdmf')
aum_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-1.pvd") # for paraview.
pvd_file << u
aum_out.close()
aum_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-2.xdmf')
aum_out.write_checkpoint(m, "m", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-2.pvd") # for paraview.
pvd_file << m
aum_out.close()

#Defining observables
h = Function(V)
h = project(a.dx(0))

pie = assemble((1/lx)*( (1-k)*(1-u**2)**2/2 + (1-k)*(u.dx(0)/kappa)**2 + (1-k)*(a*u)**2 \
                           + k*aex/2*(m.dx(0)**2) + k*m**2 - k*2*H*m \
                           + inner( a.dx(0) - H - k*m, a.dx(0) - H - k*m ) )*dx )

print("Energy density =", pie)
print("gamma =", gamma)
print("Number of iterations =", NN)
print("External Magnetic field =", H)
print("absolute tolerance =", tol)
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
c = plot(k)
plt.title(r"$k(x)$",fontsize=26)
plt.show()

