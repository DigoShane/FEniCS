#Here we solve the 2D Ginzbug Landau problem with an applied magnetic field.
#Here we want to use Energy minimization method. We start off with Gradient Descent.
#HEre a1 is \ve{A}\cdot e_1, a2 is \ve{A}\cdot e_2, u is u. However, \theta=t
#------------------------------------------------------------------------------------------------------
# For details on progress, visit the overleaf file:-
#1. Overleaf. superconductivity-Pradeep+Liping/Z3-Coding.tex/Sec. Stochastic Energy minimization methods
# /subsec. Wrote a 2D Ginzburg LAndau Stochastic Energy minimization code
#======================================================================================================
# The different stochastic methods implemented in this code are:-
# 0. Gradient Descent
# 1. Random Stochastic Gradient Descent
#    We add a noise to the gradient descent.
#    Instead of sampling noise from a sphere, I am sampling from a box.
# 2. Nesterov Momentum 
#    the gradient is evaluated at a later step. In general the moentum method saves graadients from previous steps.
# 3. Momentum method
#    v_{t+1} = \tau v_t - \eta \nabla f(x_t)
#    x_{t+1} = x_t + v_{t+1}
#The stochastic part of the update is saved in the variable {ma1,ma2,mt,mu}.
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#   b. The relaxation parameter
#   c. The absolute tolerance
#   d. The number of iterations
#   e. read_in parameter to state if we start from specified initial conditions or output of previous iteration.
#======================================================================================================
#Things to keep in mind about writing this code:-
#1. Define a functoon to evaluate the curl
#2. Define a rotation funciton.
#3. HAve replace L with l throught.
#4. All variables are lower case.
#5. REdo the code by using Hn\cdot B\perp
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
import random
import sys

#Create mesh and define function space
lx = 10
ly = 10
kappa = Constant(2.0)
mesh = RectangleMesh(Point(0., 0.), Point(lx, ly), np.ceil(lx*10/kappa), np.ceil(ly*10/kappa), "crossed")
x = SpatialCoordinate(mesh)
V = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile


# Define functions
a1 = Function(V)
a2 = Function(V)
t = Function(V)
u = Function(V)
a1_up = Function(V)
a2_up = Function(V)
t_up = Function(V)
u_up = Function(V)

# Parameters
gamma = float(input('Learning rate? -->')) # Learning rate.
NN = int(input('Number of iterations? -->')) # Number of iterations
H = Constant(input("External Magnetic field? -->"));
tol = float(input("absolute tolerance? --> "))
Ae = H*x[0] #The vec pot is A(x) = Hx_1e_2
read_in = int(input("Read from file? 1 for Yes, 0 for No --> "))
random_no_method = int(input("0 for Gradient Descent, 1 for Random Stochastic Gradient Descent, 2 for Nesterov, 3 for normal momentum --> "))
tau = float(input("tau? --> ")) # stochastic parameter
if tau < 0 or tau > 1:
  sys.exit("tau cannot be negative")

def curl(a1,a2):
    return a2.dx(0) - a1.dx(1)

#Defining the energy
Pi = ( (1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
      + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx

#Defining the gradient
Fa1 = derivative(Pi, a1)
Fa2 = derivative(Pi, a2)
Ft = derivative(Pi, t)
Fu = derivative(Pi, u)


##Setting up the initial conditions
if read_in == 0: # We want to use the standard values.
 #SC state
 print("Using bulk SC as initial condition")
 A1 = interpolate( Expression("0.0", degree=2), V)
 A2 = interpolate( Expression("0.0", degree=2), V)
 T = interpolate( Expression("1.0", degree=2), V)
 U = interpolate( Expression("1.0", degree=2), V)
 ##Modified normal state
 #print("Using modified bulk Normal as initial condition")
 #A1 = interpolate( Expression("0.0", degree=2), V)
 #A2 = interpolate( Expression("H*x[0]", H=H, degree=2), V)
 #T = interpolate( Expression("x[1]", degree=2), V)
 #U = interpolate( Expression("x[0]", degree=2), V)
 ##Vortex Solution.
 #..... need to complete
###---------------------------------------------------------------------------------------------------------------
elif read_in == 1: # We want to read from xdmf files
 #Reading input from a .xdmf file.
 print("reading in previous output as initial condition.")
 A1 = Function(V)
 A2 = Function(V)
 T = Function(V)
 U = Function(V)
 a1_in =  XDMFFile("GL-2DEnrg-0.xdmf")
 a1_in.read_checkpoint(A1,"a1",0)
 a2_in =  XDMFFile("GL-2DEnrg-1.xdmf")
 a2_in.read_checkpoint(A2,"a2",0)
 t_in =  XDMFFile("GL-2DEnrg-2.xdmf")
 t_in.read_checkpoint(T,"t",0)
 u_in =  XDMFFile("GL-2DEnrg-3.xdmf")
 u_in.read_checkpoint(U,"u",0)
 #plot(u)
 #plt.title(r"$u(x)-b4$",fontsize=26)
 #plt.show()
else:
 sys.exit("Not a valid input for read_in.")

a1_up.vector()[:] = A1.vector()[:]
a2_up.vector()[:] = A2.vector()[:]
t_up.vector()[:] = T.vector()[:]
u_up.vector()[:] = U.vector()[:]

for tt in range(NN):
 a1.vector()[:] = a1_up.vector()[:]
 a2.vector()[:] = a2_up.vector()[:]
 t.vector()[:] = t_up.vector()[:]
 u.vector()[:] = u_up.vector()[:]
 Fa1_vec = assemble(Fa1)
 Fa2_vec = assemble(Fa2)
 Ft_vec = assemble(Ft)
 Fu_vec = assemble(Fu)
 if tt == 0:
  ma1_vec = [0.0 for _ in range(len(Fa1_vec))] 
  ma2_vec = [0.0 for _ in range(len(Fa2_vec))] 
  mt_vec = [0.0 for _ in range(len(Ft_vec))] 
  mu_vec = [0.0 for _ in range(len(Fu_vec))]
 if random_no_method == 0:
  a1_up.vector()[:] = a1.vector()[:] - gamma*Fa1_vec[:]
  a2_up.vector()[:] = a2.vector()[:] - gamma*Fa2_vec[:]
  t_up.vector()[:] = t.vector()[:] - gamma*Ft_vec[:]
  u_up.vector()[:] = u.vector()[:] - gamma*Fu_vec[:]
 elif random_no_method == 1:
  ma1_vec = [random.uniform(0,tau) for _ in range(len(Fa1_vec))] 
  ma2_vec = [random.uniform(0,tau) for _ in range(len(Fa2_vec))] 
  mt_vec = [random.uniform(0,tau) for _ in range(len(Ft_vec))] 
  mu_vec = [random.uniform(0,tau) for _ in range(len(Fu_vec))]
  a1_up.vector()[:] = a1.vector()[:] - gamma*(Fa1_vec[:] + ma1_vec)
  a2_up.vector()[:] = a2.vector()[:] - gamma*(Fa2_vec[:] + ma2_vec)
  t_up.vector()[:] = t.vector()[:] - gamma*(Ft_vec[:] + mt_vec)
  u_up.vector()[:] = u.vector()[:] - gamma*(Fu_vec[:] + mu_vec)
 elif random_no_method == 3:
  #print("tt=",tt," type of vec =",type(ma1_vec))
  ma1_vec = [ii * tau for ii in ma1_vec]
  ma1_vec[:] = ma1_vec[:] - gamma*Fa1_vec[:] 
  a1_up.vector()[:] = a1.vector()[:] + ma1_vec
  ma2_vec = [ii * tau for ii in ma2_vec]
  ma2_vec[:] = ma2_vec[:] - gamma*Fa2_vec[:] 
  a2_up.vector()[:] = a2.vector()[:] + ma2_vec
  mt_vec = [ii * tau for ii in mt_vec]
  mt_vec[:] = mt_vec[:] - gamma*Ft_vec[:] 
  t_up.vector()[:] = t.vector()[:] + mt_vec
  mu_vec = [ii * tau for ii in mu_vec]
  mu_vec[:] = mu_vec[:] - gamma*Fu_vec[:] 
  u_up.vector()[:] = u.vector()[:] + mu_vec
 elif random_no_method == 2:
   sys.exit("Not ready yet. Nesterov requires gradient at next iteration. Not sure how to supply that.")
 else:
  sys.exit("Not a valid input for random_no_method.")
 #print(Fa1_vec.get_local()) # prints the vector.
 #print(np.linalg.norm(np.asarray(Fa1_vec.get_local()))) # prints the vector's norm.
 tol_test = np.linalg.norm(np.asarray(Fa1_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fa2_vec.get_local()))\
           +np.linalg.norm(np.asarray(Ft_vec.get_local()))\
           +np.linalg.norm(np.asarray(Fu_vec.get_local()))
 print(tol_test)
 if float(tol_test)  < tol :
  break
 

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
a1a2tu_out.write_checkpoint(t, "t", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-2.pvd") # for paraview.
pvd_file << t
a1a2tu_out.close()
a1a2tu_out = XDMFFile('GL-2DEnrg-3.xdmf')
a1a2tu_out.write_checkpoint(u, "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
pvd_file = File("GL-2DEnrg-3.pvd") 
pvd_file << u
a1a2tu_out.close()


pie = assemble((1/(lx*ly))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) \
                        + ( (a1-t.dx(0))**2 + (a2-t.dx(1))**2 )*u**2 \
                            + inner( curl(a1 ,a2-Ae), curl(a1 ,a2-Ae) ) )*dx )
print("Energy density =", pie)


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
c = plot(t)
plt.title(r"$\theta(x)$",fontsize=26)
plt.colorbar(c)
plt.show()

