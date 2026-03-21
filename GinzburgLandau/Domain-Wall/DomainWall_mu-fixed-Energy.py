#This is the same code as DomainWall-mu-fixed.py but with we are using the energetic approach.
#------------------------------------------------------------------------------------------------------
# For details on progress, you need to check two files. For the Theory of why we can map it to Ginzburg Landau check UH One note/superconductivity/ 1D Gurtin Tensor solution/ Domain Wall-3 
# For the weak formulation for FEniCS, check UH One note/superconductivity/ 1D Gurtin Tensor solution/ Domain Wall-2.
# ALternatively, you can check UH One note/superconductivity/ Coding/ 1D Ginzburg Landau fenics.
# Especially section V.
#======================================================================================================
#We just borrowed the code form Desktop/Ubuntu-Codes/FEniCS/GinzburgLandau/1D/GinzburgLandau-1D-Constraint4.py


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

#Check if tensor is positive definite
def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


#parameters
K11 = float(1.0)  # float(input("Enter K11: "))
K12 = float(0.1)  # float(input("Enter K12: "))
K22 = float(4.0)  # float(input("Enter K22: "))
T11 = float(0.4)  # float(input("Enter T11: "))
T12 = float(-0.1)  # float(input("Enter T12: "))
T21 = float(-0.1)  # float(input("Enter T21: "))
T22 = float(0.6)  # float(input("Enter T22: "))
H = float(np.sqrt(2)) # float(input("Enter ext field : ")) 
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
tol = float(input("absolute tolerance? --> "))
f = float(input("Enter fraction of 2\pi? --> "))

x = np.array([[K11,K12,T11,T21],[K12,K22,T12,T22],[T11,T12,1.0,0.0],[T21,T22,0.0,1.0]])
print(is_pos_def(x))
if is_pos_def(x):
    print("Gamma is positive definite.")
else:
    sys.exit("Gamma is not positive definite. Please re-enter K_m,T_m,M_m.")


#Create mesh and define function space
Lx = 50
kappa = max(K11, K22)
mesh = fe.IntervalMesh( int(np.ceil(Lx*10/kappa)),0,Lx)
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile

# Define functions
a = Function(V)
u = Function(V)
a_up = Function(V)
u_up = Function(V)

#setting Dirichlet BC 
def boundary_L(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], 0, tol)
def boundary_R(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], Lx, tol)

mu = float(f*np.pi) 
d1 = np.cos(mu)
d2 = np.sin(mu)
dKd = float(K11*d1**2 + K22*d2**2 + 2*K12*d1*d2) 
T2 = float(T11*d1**2 + T22*d2**2 + (T12+T21)*d1*d2) 
T1 = float(T21*d1**2 - T12*d2**2 + (T22-T11)*d1*d2) 
K1 = float(dKd-T2*T2) 

print("dKd, T2, T1, K1, K1-T1*T1 = ", dKd, T2, T1, K1, K1-T1*T1)

#Defining the energy
Pi = ( (1-u**2)**2/2 + K1*inner(grad(u), grad(u)) + 2*u*u.dx(0)*T1*a + a**2*u**2 \
      + (a.dx(0) - H)**2 + 100*(2*u-1)**2 )*dx

#Defining the gradient
Fa = derivative(Pi, a)
Fu = derivative(Pi, u)


##Setting up the initial conditions
##Coexistence of phase as initial condition
#ul = Expression('0', degree=2, domain=mesh)
#Al = Expression('H*(0.5*Lx-x[0])', H=H, Lx=Lx, degree=2, domain=mesh)
#ur = Expression('1', degree=2, domain=mesh)
#Ar = Expression('0', degree=2, domain=mesh)
#SC phase as initial condition
U = interpolate( Expression('1', degree=2), V)
A = interpolate( Expression('0', H=H, Lx=Lx, degree=2), V)


a_up.vector()[:] = A.vector()[:]
u_up.vector()[:] = U.vector()[:]

for tt in range(NN):
 a.vector()[:] = a_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]
 Fa_vec = assemble(Fa)
 Fu_vec = assemble(Fu)
 a_up.vector()[:] = a.vector()[:] - gamma*Fa_vec[:]
 u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]
 #print(Fa1_vec.get_local()) # prints the vector.
 #print(np.linalg.norm(np.asarray(Fa1_vec.get_local()))) # prints the vector's norm.
 tol_test = np.linalg.norm(np.asarray(Fa_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fu_vec.get_local()))
 print(tol_test)
 if float(tol_test)  < tol :
  break
 

pie = assemble( (1/Lx)*( (1-u**2)**2/2 + K1*inner(grad(u), grad(u)) + 2*T1*u*A*u.dx(0) \
        + u*u*A*A + (A.dx(0)-H)**2 )*dx - 0.5)


print("surface energy  for \mu=", mu," is ", pie)

fig=plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
fig=plot(a)
plt.title(r"$A(x)$",fontsize=26)
plt.show()


