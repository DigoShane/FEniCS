#Here we solve the 2D Ginzbug Landau problem with an applied magnetic field and circular magnetic inhomogeneity.
#The problem is assumed 2D.
#Here we want to use Energy minimization method. We start off with Gradient Descent.
#HEre a1 is \ve{A}\cdot e_1, a2 is \ve{A}\cdot e_2, fR is \Re(\psi) and fI is \Im(\psi). m1, m2 and m3 are the components of the magnetization of the inhomogeneity.
#---------------------------------------------------------------------------------------------------------------
# The energy functional in non-dimensional form is presented below. For the derivation see "Overleaf.Superconductivity-Pradeep+Liping/Z7-MAgnet+Vortex.tex
# "/Section.Non-dimensopnal Complex valued Minimization problem".
# \int_{\Omega_s} (1-|f|^2)^2/2 + |i\nabla f/\kappa+Af|^2 dx +\int_{\Omega_m} aex/2(\nabla m)^2 + \phi(m) - 2H.m dx + \int_{\Scr{R}^3} |\curl A - H - m\mathbbm{1}_{\Omega_m}|^2 dx
#where f is the complex valued normalized SC order parameter. \Omega_S is the superconducting domain and \Omega_m is the magnetic inhomogeneity domain.
# We can will seperate f into its real and imaginary parts as fR and fI respectively. 
# the 2nd term becomes:
#  |i\nabla f/\kappa+Af|^2 = (A*fR-\nabla fI/\kappa)^2 + (A*fI+\nabla fR/\kappa)^2
# Further, since \ve{m}=m\ve{e}_3.
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
ly = float(20)
kappa = Constant(2.0)
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), np.ceil(lx*10/kappa), np.ceil(ly*10/kappa), "crossed")
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile
tol_prev = 1 # Initializing tol_prev to 1.


# Define functions
a1 = Function(V)
a2 = Function(V)
fR = Function(V)
fI = Function(V)
m = Function(V)
a1_up = Function(V)
a2_up = Function(V)
fR_up = Function(V)
fI_up = Function(V)
m_up = Function(V)

# Parameters
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = float(input("External Magnetic field? -->"));
tol = float(0.000001) #float(input("absolute tolerance? --> "))
R = float(input("Radius of the inhomogeneity? --> "));
aex = float(input("Exchange parameter? --> "));
d = float(input("Vortex offset? --> "));
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

#Defining the characteristic function.
Q = FunctionSpace(mesh, 'DG', 0)
k = Function(Q)
kc = Function(Q)
k = interpolate( Expression("(x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly) <= R*R ? 1 : 0", lx=lx, ly=ly, R=R, degree=2), Q)
kc = interpolate( Expression("(x[0]-0.5*lx)*(x[0]-0.5*lx)+(x[1]-0.5*ly)*(x[1]-0.5*ly) <= R*R ? 0 : 1", lx=lx, ly=ly, R=R, degree=2), Q)

#Defining the energy
Pi = ( kc*(1-fR**2-fI**2)**2/2 + kc*(a1*fR-fI.dx(0)/kappa)**2 + kc*(a2*fR-fI.dx(1)/kappa)**2 \
      + kc*(a1*fI+fR.dx(0)/kappa)**2 + kc*(a2*fI+fR.dx(1)/kappa)**2 \
      + k*aex/2*(m.dx(0)**2+m.dx(1)**2) + k*m**2 - 2*H*m*k \
      + inner( curl(a1 ,a2) - H - k*m, curl(a1 ,a2) - H - k*m ) )*dx


#Defining the gradient
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
FfR = derivative(Pi, fR)
FfI = derivative(Pi, fI)
Fm = derivative(Pi, m)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 ##SC state
 #print("Using bulk SC as initial condition")
 #A1 = interpolate( Expression("0.0", degree=2), V)
 #A2 = interpolate( Expression("0.0", degree=2), V)
 #FR = interpolate( Expression("1.0", degree=2), V)
 #FI = interpolate( Expression("1.0", degree=2), V)
 #M = interpolate( Expression("0.0", degree=2), V)
 #Vortex Solution.
 print("Using Vortex solution")
 A1 = interpolate( Expression('sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? -x[1] : \
                             -exp(-sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *x[1]/sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
                                lx=lx, ly=ly, r=0.3517,d=d, K=kappa, degree=1), V)
 A2 = interpolate( Expression('sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? x[0] : \
                             exp(-sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *(x[0]-0.5*lx-d)/sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly))*1/K', \
                                lx=lx, ly=ly, r=0.3517,d=d, K=kappa, degree=1), V)
 FR = interpolate( Expression('sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? 0 : \
                             tanh(sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *(x[0]-0.5*lx-d)/sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly))', \
                                lx=lx, ly=ly, r=0.05,d=d, degree=1), V)
 FI = interpolate( Expression('sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly)) <= r + DOLFIN_EPS ? 0 : \
                             tanh(sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly))) \
                              *(x[1]-0.5*ly)/sqrt((x[0]-0.5*lx-d)*(x[0]-0.5*lx-d)+(x[1]-0.5*ly)*(x[1]-0.5*ly))', \
                                lx=lx, ly=ly, r=0.05,d=d, degree=1), V)
 M = interpolate( Expression("0.0", degree=2), V)
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 FR = Function(V)
 FI = Function(V)
 M = Function(V)
 a1_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 fR_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-2.xdmf")
 fR_in.read_checkpoint(FR,"fR",0)
 fI_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-3.xdmf")
 fI_in.read_checkpoint(FI,"fI",0)
 m_in =  XDMFFile("GL-Magnet+Vortex-2DEnrg-4.xdmf")
 m_in.read_checkpoint(M,"m",0)
else:
 import sys
 sys.exit("Not a valid input for read_in.")

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
fR_up.vector()[:] = FR.vector()[:]
fI_up.vector()[:] = FI.vector()[:]
m_up.vector()[:]  = M.vector()[:]

for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 fR.vector()[:] = fR_up.vector()[:]
 fI.vector()[:] = fI_up.vector()[:]
 m.vector()[:]  = m_up.vector()[:]

 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 FfR_vec = assemble(FfR)
 FfI_vec = assemble(FfI)
 Fm_vec  = assemble(Fm)

 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 fR_up.vector()[:] = fR.vector()[:] - gamma*FfR_vec[:]
 fI_up.vector()[:] = fI.vector()[:] - gamma*FfI_vec[:]
 m_up.vector()[:] = m.vector()[:] - gamma*Fm_vec[:]
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(FfR_vec.get_local()))\
           +np.linalg.norm(np.asarray(FfI_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fm_vec.get_local()))
 #print(tol_test)
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
 
print(tol_test)

##Save solution in a .xdmf file and for paraview.
a1a2tu_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-0.xdmf')
a1a2tu_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-0.pvd") # for paraview. 
pvd_file << a1
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-1.xdmf')
a1a2tu_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-1.pvd") # for paraview. 
pvd_file << a2
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-2.xdmf')
a1a2tu_out.write_checkpoint(fR, "fR", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-2.pvd") # for paraview.
pvd_file << fR
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-3.xdmf')
a1a2tu_out.write_checkpoint(fI, "fI", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-3.pvd") # for paraview.
pvd_file << fI
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-Magnet+Vortex-2DEnrg-4.xdmf')
a1a2tu_out.write_checkpoint(m, "m", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-Magnet+Vortex-2DEnrg-4.pvd") 
pvd_file << m
a1a2tu_out.close()

#Defining observables
h = Function(V)
u = Function(V)
h = project(curl(a1 ,a2))
u = project(fR**2 + fI**2)

pie = assemble((1/(lx*ly))*( (1-k)*(1-fR**2-fI**2)**2/2 + (1-k)*(a1*fR-fI.dx(0)/kappa)**2 + (1-k)*(a2*fR-fI.dx(1)/kappa)**2 \
                           + (1-k)*(a1*fI+fR.dx(0)/kappa)**2 + (1-k)*(a2*fI+fR.dx(1)/kappa)**2 \
                           + k*aex/2*(m.dx(0)**2+m.dx(1)**2) + k*m**2 - 2*H*m*k \
                           + inner( curl(a1 ,a2) - H -k*m, curl(a1 ,a2) - H -k*m ) )*dx )

print("Energy density =", pie)
print("gamma =", gamma)
print("Number of iterations =", NN)
print("External Magnetic field =", H)
print("absolute tolerance =", tol)
print("Radius of the inhomogeneity =", R)
print("Exchange parameter =", aex)
print("Read from file? 1 for Yes, 0 for No =", read_in)

c = plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(a1)
plt.title(r"$A_1(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(a2)
plt.title(r"$A_2(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(h)
plt.title(r"$h(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(m)
plt.title(r"$m(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(k)
plt.title(r"$k(x)$",fontsize=26)
plt.colorbar(c)
plt.show()
c = plot(kc)
plt.title(r"$k^c(x)$",fontsize=26)
plt.colorbar(c)
plt.show()

