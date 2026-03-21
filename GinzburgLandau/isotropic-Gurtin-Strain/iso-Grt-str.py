#Here we solve the 2D Ginzbug Landau Gurtin mod with strain coupling.
# We have externally applied stress and magnetic field.
#Here we want to use Energy minimization method. We start off with Gradient Descent.
#HEre a1 is \ve{A}\cdot e_1, a2 is \ve{A}\cdot e_2, u is u. However, \theta=t
#------------------------------------------------------------------------------------------------------
# For details on progress, visit the overleaf file:-
#1. Overleaf. superconductivity-Pradeep+Liping/Z3-Coding.tex/Sec. Stochastic Energy minimization methods
# /subsec. Gradient Descent in FEniCS/paragraph{Wrote a 2D Ginzburg LAndau Energy minimization code}
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#   b. The relaxation parameter
#   c. The absolute tolerance
#2. When reading from and writing into respective files,
#   we are writing the lagrange multiplier as a constant function
#   When reading the functions, we interpolate onto a space VAu.
#======================================================================================================
#Things to keep in mind about writing this code:-
#1. Define a functoon to evaluate the curl
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


#Create mesh and define function space
lx = 10
ly = 10
lz = 10
kappa = Constant(2.0)
N = 100
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(lx, ly, lz), N, N, N)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile


# Define functions
a1 = Function(V) #A_1 --> magnetic vector potential
a2 = Function(V) #A_2
a3 = Function(V) #A_3
v1 = Function(V) #v_1 --> displacement
v2 = Function(V) #v_2
v3 = Function(V) #v_3
fR = Function(V) #Re(\psi)
fI = Function(V) #Imag(\psi)
a1_up = Function(V)
a2_up = Function(V)
a3_up = Function(V)
v1_up = Function(V)
v2_up = Function(V)
v3_up = Function(V)
fR_up = Function(V)
fI_up = Function(V)

# Parameters
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H1 = Constant(0.0) #Constant(input("H1? -->"));
H2 = Constant(0.0) #Constant(input("H2? -->"));
H3 = Constant(0.7) #Constant(input("H3? -->"));
tau = Constant(0.5) #Constant(input("tau? -->"));
lmbda = Constant(1.0) #Constant(input("Lame parameter lambda? -->"));
mu = Constant(2.0) #Constant(input("Lame parameter mu? -->"));
Gamma1 = Constant(1.0) # Constant(input("gamma1? -->")); #Gamma is \gamma in the notes. To distinguish from learning rate.
Gamma2 = Constant(1.0) #Constant(input("gamma2? -->"));
Gamma3 = Constant(0.0) #Constant(input("gamma3? -->"));
s11 = Constant(1.0) #Constant(input("s11? -->")); #s is the externally applied stress.
s12 = Constant(1.0) #Constant(input("s12? -->"));
s13 = Constant(0.0) #Constant(input("s13? -->"));
s23 = Constant(0.0) #Constant(input("s23? -->"));
s22 = Constant(1.0) #Constant(input("s22? -->"));
s33 = Constant(0.0) #Constant(input("s33? -->"));
tol = float(input("absolute tolerance? --> "))
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))

def ssqrt(x):
    return sqrt(abs(x)+ 1e-6)

def slog(x):
    return ln(abs(x)+ 1e-6)

def uf(fR,fI):
    return ssqrt(fR**2+fI**2)

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

def eps11(v1):
    return v1.dx(0)

def eps22(v2):
    return v2.dx(1)

def eps33(v3):
    return v3.dx(2)

def eps12(v1,v2):
    return 0.5*(v1.dx(1) + v2.dx(0))

def eps13(v1,v3):
    return 0.5*(v3.dx(0) + v1.dx(2))

def eps23(v2,v3):
    return 0.5*(v2.dx(2) + v3.dx(1))

def A0u1(a1,fR,fI):
    return a1*uf(fR,fI) + fI*fR.dx(0)-fR*fI.dx(0)

def A0u2(a2,fR,fI):
    return a2*uf(fR,fI) + fI*fR.dx(1)-fR*fI.dx(1)

def A0u3(a3,fR,fI):
    return a3*uf(fR,fI) + fI*fR.dx(2)-fR*fI.dx(2)



#Defining the energy
Pi = ( (1-fR**2-fI**2)**2/2 + ( fR.dx(0)/kappa + a1*fR )**2 + ( fR.dx(1)/kappa + a2*fR )**2  + ( fR.dx(2)/kappa + a3*fR )\
      + ( fI.dx(0)/kappa + a1*fI )**2 + ( fI.dx(1)/kappa + a2*fI )**2 + ( fI.dx(2)/kappa + a3*fI )**2 \
      + 2*tau*(slog(uf(fR,fI)).dx(0))*(a1*(fR**2+fI**2) + (fI*fR.dx(0) - fR*fI.dx(0))/kappa) \
      + 2*tau*(slog(uf(fR,fI)).dx(1))*(a2*(fR**2+fI**2) + (fI*fR.dx(1) - fR*fI.dx(1))/kappa) \
      + 2*tau*(slog(uf(fR,fI)).dx(2))*(a3*(fR**2+fI**2) + (fI*fR.dx(2) - fR*fI.dx(2))/kappa) \
      + 0.5*lmbda*( eps11(v1) + eps22(v2) + eps33(v3))**2 + mu*( eps11(v1)**2 + eps22(v2)**2 + eps33(v3)**2 + 2*eps12(v1,v2)**2 + 2*eps13(v1,v3)**2 + 2*eps23(v2,v3)**2 ) \
      - ( s11*eps11(v1) + s22*eps22(v2) + s33*eps33(v3) + 2*s12*eps12(v1,v2) + 2*s13*eps13(v1,v3) + 2*s23*eps23(v2,v3) ) \
      + Gamma1*( eps11(v1)*A0u1(a1,fR,fI) + eps12(v1,v2)*A0u2(a2,fR,fI) + eps13(v1,v3)*A0u3(a3,fR,fI) ) \
      + Gamma2*( eps12(v1,v2)*A0u1(a1,fR,fI) + eps22(v2)*A0u2(a2,fR,fI) + eps23(v2,v3)*A0u3(a3,fR,fI) ) \
      + Gamma3*( eps13(v1,v3)*A0u1(a1,fR,fI) + eps23(v2,v3)*A0u2(a2,fR,fI) + eps33(v3)*A0u3(a3,fR,fI) ) \
      + (a3.dx(1)-a2.dx(2) - H1)**2 + (a1.dx(2)-a3.dx(0) - H2)**2 + (a2.dx(0)-a1.dx(1) - H3)**2)*dx

#Defining the gradient
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
Fa3 = derivative(Pi, a3)
Fv1 = derivative(Pi, v1)
Fv2 = derivative(Pi, v2)
Fv3 = derivative(Pi, v3)
FfR = derivative(Pi, fR)
FfI = derivative(Pi, fI)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 #SC state
 print("Using bulk SC as initial condition")
 A1 = interpolate( Expression("0.0", degree=2), V)
 A2 = interpolate( Expression("0.0", degree=2), V)
 A3 = interpolate( Expression("0.0", degree=2), V)
 V1 = interpolate( Expression("0.0", degree=2), V)
 V2 = interpolate( Expression("0.0", degree=2), V)
 V3 = interpolate( Expression("0.0", degree=2), V)
 FR = interpolate( Expression("1.0", degree=2), V)
 FI = interpolate( Expression("1.0", degree=2), V)
 ##Vortex Solution.
 #..... need to complete
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 A3 = Function(V)
 V1 = Function(V)
 V2 = Function(V)
 V3 = Function(V)
 FR = Function(V)
 FI = Function(V)
 a1_in =  XDMFFile("GL-2DEnrg-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in =  XDMFFile("GL-2DEnrg-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 a3_in =  XDMFFile("GL-2DEnrg-2.xdmf")
 a3_in.read_checkpoint(A3,"a3",0)
 v1_in =  XDMFFile("GL-2DEnrg-3.xdmf")
 v1_in.read_checkpoint(V1,"v1",0)
 v2_in =  XDMFFile("GL-2DEnrg-4.xdmf")
 v2_in.read_checkpoint(V2,"v2",0)
 v3_in =  XDMFFile("GL-2DEnrg-5.xdmf")
 v3_in.read_checkpoint(V3,"v3",0)
 fR_in =  XDMFFile("GL-2DEnrg-6.xdmf")
 fR_in.read_checkpoint(FR,"fR",0)
 fI_in =  XDMFFile("GL-2DEnrg-7.xdmf")
 fI_in.read_checkpoint(FI,"fI",0)
else:
 import sys
 sys.exit("Not a valid input for read_in.")

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
a3_up.vector()[:] = A3.vector()[:]
v1_up.vector()[:] = V1.vector()[:]
v2_up.vector()[:] = V2.vector()[:]
v3_up.vector()[:] = V3.vector()[:]
fR_up.vector()[:] = FR.vector()[:]
fI_up.vector()[:] = FI.vector()[:]

for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 a3.vector()[:] = a3_up.vector()[:]
 v1.vector()[:] = v1_up.vector()[:]
 v2.vector()[:] = v2_up.vector()[:]
 v3.vector()[:] = v3_up.vector()[:]
 fR.vector()[:] = fR_up.vector()[:]
 fI.vector()[:] = fI_up.vector()[:]
 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 Fa3_vec = assemble(Fa3)
 Fv1_vec = assemble(Fv1)
 Fv2_vec = assemble(Fv2)
 Fv3_vec = assemble(Fv3)
 FfR_vec = assemble(FfR)
 FfI_vec = assemble(FfI)
 a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
 a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
 a3_up.vector()[:] = a3.vector()[:] - gamma*Fa3_vec[:]
 v1_up.vector()[:] = v1.vector()[:] - gamma*Fv1_vec[:]
 v2_up.vector()[:] = v2.vector()[:] - gamma*Fv2_vec[:]
 v3_up.vector()[:] = v3.vector()[:] - gamma*Fv3_vec[:]
 fR_up.vector()[:] = fR.vector()[:] - gamma*FfR_vec[:]
 fI_up.vector()[:] = fI.vector()[:] - gamma*FfI_vec[:]
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa3_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fv1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fv2_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fv3_vec.get_local()))\
           +np.linalg.norm(np.asarray(FfR_vec.get_local()))\
           +np.linalg.norm(np.asarray(FfI_vec.get_local()))
 print(tol_test)
 if float(tol_test)  < tol :
  break
 

##Save solution in a .xdmf file and for paraview.
avf_out = XDMFFile('GL-2DEnrg-0.xdmf')
avf_out.write_checkpoint(a1, "a1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-0.pvd") # for paraview. 
pvd_file << a1
avf_out.close()

avf_out = XDMFFile('GL-2DEnrg-1.xdmf')
avf_out.write_checkpoint(a2, "a2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-1.pvd") # for paraview. 
pvd_file << a2
avf_out.close()

avf_out = XDMFFile('GL-2DEnrg-2.xdmf')
avf_out.write_checkpoint(a3, "a3", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-2.pvd") # for paraview. 
pvd_file << a3
avf_out.close()

avf_out = XDMFFile('GL-2DEnrg-3.xdmf')
avf_out.write_checkpoint(v1, "v1", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-3.pvd") # for paraview. 
pvd_file << v1
avf_out.close()

avf_out = XDMFFile('GL-2DEnrg-4.xdmf')
avf_out.write_checkpoint(v2, "v2", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-4.pvd") # for paraview. 
pvd_file << v2
avf_out.close()

avf_out = XDMFFile('GL-2DEnrg-5.xdmf')
avf_out.write_checkpoint(v3, "v3", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-5.pvd") # for paraview. 
pvd_file << v3
avf_out.close()

avf_out = XDMFFile('GL-2DEnrg-6.xdmf')
avf_out.write_checkpoint(fR, "fR", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-6.pvd") # for paraview.
pvd_file << fR
avf_out.close()

avf_out = XDMFFile('GL-2DEnrg-7.xdmf')
avf_out.write_checkpoint(fI, "fI", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-7.pvd") 
pvd_file << fI
avf_out.close()


pie = assemble((1/(lx*ly))*( (1-fR**2-fI**2)**2/2 + ( fR.dx(0)/kappa + a1*fR )**2 + ( fR.dx(1)/kappa + a2*fR )**2  + ( fR.dx(2)/kappa + a3*fR )\
      + ( fI.dx(0)/kappa + a1*fI )**2 + ( fI.dx(1)/kappa + a2*fI )**2 + ( fI.dx(2)/kappa + a3*fI )**2 \
      + 2*tau*(slog(uf(fR,fI)).dx(0))*(a1*(fR**2+fI**2) + (fI*fR.dx(0) - fR*fI.dx(0))/kappa) \
      + 2*tau*(slog(uf(fR,fI)).dx(1))*(a2*(fR**2+fI**2) + (fI*fR.dx(1) - fR*fI.dx(1))/kappa) \
      + 2*tau*(slog(uf(fR,fI)).dx(2))*(a3*(fR**2+fI**2) + (fI*fR.dx(2) - fR*fI.dx(2))/kappa) \
      + 0.5*lmbda*( eps11(v1) + eps22(v2) + eps33(v3))**2 + mu*( eps11(v1)**2 + eps22(v2)**2 + eps33(v3)**2 + 2*eps12(v1,v2)**2 + 2*eps13(v1,v3)**2 + 2*eps23(v2,v3)**2 ) \
      - s11*eps11(v1) + s22*eps22(v2) + s33*eps33(v3) + 2*s12*eps12(v1,v2) + 2*s13*eps13(v1,v3) + 2*s23*eps23(v2,v3) \
      + Gamma1*( eps11(v1)*A0u1(a1,fR,fI) + eps12(v1,v2)*A0u2(a2,fR,fI) + eps13(v1,v3)*A0u3(a3,fR,fI) ) \
      + Gamma2*( eps12(v1)*A0u1(a1,fR,fI) + eps22(v1,v2)*A0u2(a2,fR,fI) + eps23(v1,v3)*A0u3(a3,fR,fI) ) \
      + Gamma3*( eps13(v1)*A0u1(a1,fR,fI) + eps23(v1,v2)*A0u2(a2,fR,fI) + eps33(v1,v3)*A0u3(a3,fR,fI) ) \
      + (a3.dx(1)-a2.dx(2) - H1)**2 + (a1.dx(2)-a3.dx(0) - H2)**2 + (a2.dx(0)-a1.dx(1) - H3)**2)*dx )



print("Energy density =", pie)


c = plot(fR)
plt.title(r"$fR(x)$",fontsize=26)
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
c = plot(a3)
plt.title(r"$A_3(x)$",fontsize=26)
plt.colorbar(c)
plt.show()

