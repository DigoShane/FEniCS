#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
#Here we want to use the constrained domain wall formulaton in 1D.
#HEre a1 is \ve{A}\cdot e_1, a2 is \ve{A}\cdot e_2, u is u. However, \theta=t
#------------------------------------------------------------------------------------------------------
# For details on progress, you need to check two files. For the Theroy of why we can map it to Ginzburg Landau check UH One note/superconductivity/ 1D Gurtin Tensor solution/ Domain Wall-3 
# For the weak formulation for FEniCS, check UH One note/superconductivity/ 1D Gurtin Tensor solution/ Domain Wall-2.
# ALternatively, you can check UH One note/superconductivity/ Coding/ 1D Ginzburg Landau fenics.
# Especially section V.
#======================================================================================================
#WE just borrowed the code form Desktop/Ubuntu-Codes/FEniCS/GinzburgLandau/1D/GinzburgLandau-1D-Constraint4.py


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
H = float(np.sqrt(2))
step = float(input("Enter step size : ")) #float(0.1)
gamma = float(0.01) #float(input('Learning rate? -->')) # Learning rate.
#NN = int(input('Number of iterations? -->')) # Number of iterations
#read_in = int(input('1 for new values, 0 for previous? -->')) # Number of iterations
rel_tol = float(input('What is the relative tolerance? -->')) # Number of iterations
tol_abs = float(input("absolute tolerance? --> "))
rlx_par = float(input("Relaxation parameter? --> "))

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
VA = FiniteElement("CG", mesh.ufl_cell(), 2)
Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
R = FiniteElement("Real", mesh.ufl_cell(), 0)
V = FunctionSpace(mesh, MixedElement(VA, Vu, R))
RFnSp = FunctionSpace(mesh, "Real", 0)
Vcoord = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile

# Define functions
dAur = TrialFunction(V)
(dA, du, dr) = split(dAur)

#setting Dirichlet BC 
def boundary_L(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], 0, tol)
def boundary_R(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], Lx, tol)

bc1 = DirichletBC(V.sub(1), 0, boundary_L)
bc2 = DirichletBC(V.sub(1), 1, boundary_R)
bc3 = DirichletBC(V.sub(0), 0, boundary_R)
bcs = [bc1, bc2, bc3];

mu_values = np.arange(0, 2*np.pi, step)
surface_energies = []
plot_list = np.arange(0, 2*np.pi, np.pi/4)

for mu in mu_values:
   
    d1 = np.cos(mu)
    d2 = np.sin(mu)
    dKd = float(K11*d1**2 + K22*d2**2 + 2*K12*d1*d2) 
    T2 = float(T11*d1**2 + T22*d2**2 + (T12+T21)*d1*d2) 
    T1 = float(T21*d1**2 - T12*d2**2 + (T22-T11)*d1*d2) 
    K1 = float(dKd-T2*T2) 
 
    ##Setting up the initial conditions
    if mu == 0: # We want to use the standard values.
     #Coexistence of phase as initial condition
     ul = Expression('0', degree=2, domain=mesh)
     Al = Expression('H*(0.5*Lx-x[0])', H=H, Lx=Lx, degree=2, domain=mesh)
     ur = Expression('1', degree=2, domain=mesh)
     Ar = Expression('0', degree=2, domain=mesh)
     Aur = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur', '11'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
    ###---------------------------------------------------------------------------------------------------------------
    else: # We want to read from xdmf files
     #Reading input from a .xdmf file.
     Aur = Function(V)
     A = Function(Vcoord)
     u = Function(Vcoord)
     r = Function(RFnSp)
     data = np.loadtxt('test-2-Constraint4.txt')
     y0 = data
     r = interpolate(Constant(float(y0)),RFnSp)
     A_in =  XDMFFile("test-0-Constraint4.xdmf")
     A_in.read_checkpoint(A,"A",0)
     u_in =  XDMFFile("test-1-Constraint4.xdmf")
     u_in.read_checkpoint(u,"u",0)
     assign(Aur,[A,u,r])

    (A, u, r) = split(Aur)
    
    F = ( -(1-u**2)*u*du + K1*inner(grad(u), grad(du)) - T1*A.dx(0)*u*du + A**2*u*du + r*du \
        + dr*(2*u-1) + inner(grad(A), grad(dA)) + u**2*A*dA )*dx + H*dA*ds
    solve(F == 0, Aur, bcs,
       solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.001,"absolute_tolerance":tol_abs,"maximum_iterations":5000}})
    
    A = Aur.sub(0, deepcopy=True)
    u = Aur.sub(1, deepcopy=True)
    r = Aur.sub(2, deepcopy=True)

    ##Save solution in a .xdmf file and for paraview.
    Aur_split = Aur.split(True)
    Aur_out = XDMFFile('test-0-Constraint4.xdmf')
    Aur_out.write_checkpoint(Aur_split[0], "A", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
    Aur_out.close()
    Aur_out = XDMFFile('test-1-Constraint4.xdmf')
    Aur_out.write_checkpoint(Aur_split[1], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
    Aur_out.close()
    with open("test-2-Constraint4.txt", "w") as file:
        print(float(Aur_split[2]), file=file)

    if any(abs(mu - plot) <= step for plot in plot_list) :
       fig=plot(u)
       plt.title(r"$u(x)$ for $\mu=$%3.3f, H=%3.3f"%mu,%H,fontsize=26)
       plt.show()
       fig=plot(A)
       plt.title(r"$A(x)$ for $\mu=$%3.3f, H=%3.3f"%mu,%H,fontsize=26)
       plt.show()

    pie = assemble( (1/Lx)*( (1-u**2)**2/2 + K1*inner(grad(u), grad(u)) + 2*T1*u*A*u.dx(0) \
            + u*u*A*A + (A.dx(0)-H)**2 )*dx )
    surface_energies.append(pie)


#Plot
plt.plot(mu_values, surface_energies)
plt.xlabel(r"$d(\mu)$",fontsize=18)
plt.ylabel(r"$\sigma$",fontsize=18)
plt.show()



