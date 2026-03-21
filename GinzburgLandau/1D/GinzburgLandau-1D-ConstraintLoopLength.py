#Here we solve the 1D Ginzbug Landau problem with an applied critical field, H=1/√2.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/Ginsburg landau fenicsi/sec-V.
#This is a modification of the original code in that we have added an integral constraint.
#------------------------------------------------------------------------------------------------------
#HOW IS THIS CODE DIFFERENT:-
#1. The way this code runs is that I run it for each loop
#



from dolfin import *
import fenics as fe
import numpy as np
from ufl import tanh
import matplotlib.pyplot as plt

# Optimization options for the form compiler
parameters["krylov_solver"]["nonzero_initial_guess"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Parameters
kappa = Constant(1);
H = Constant('0.7071');#Constant(Hin);
rlx_par = Constant("1");#Constant(rlx_par_in);
tol_abs = Constant("0.000001");#Constant(tol_abs_in);
nn = 100;#No. of plots to skip over


#Create mesh and define function space
for ii in range(10,501,1):
 Lx=float(ii)
 mesh = fe.IntervalMesh(int(Lx)*10,0,Lx)
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
 
 Ae = H*x[0]
 
 #Reading input from a .xdmf file.
 Aur = Function(V)
 A = Function(Vcoord)
 u = Function(Vcoord)
 r = Function(RFnSp)
 data = np.loadtxt('test-2-ConstraintLoopLength.txt')
 y0 = data
 r = interpolate(Constant(float(y0)),RFnSp)
 A_in =  XDMFFile("test-0-ConstraintLoopLength.xdmf")
 A_in.read_checkpoint(A,"A",0)
 u_in =  XDMFFile("test-1-ConstraintLoopLength.xdmf")
 u_in.read_checkpoint(u,"u",0)
 assign(Aur,[A,u,r])
 
 (A, u, r) = split(Aur)
 
 if ii%int(nn) == 0:
  plot(u)
  plt.title(r"$u(x)-b4$",fontsize=26)
  plt.show()
  plot(A)
  plt.title(r"$A(x)e_2-b4$",fontsize=26)
  plt.show()
 
 
 
 F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + 0.5*r*du + dr*(u-0.5) + u**2*A*dA + inner(grad(A), grad(dA)))*dx + H*dA*ds#testing with original problem.
 solve(F == 0, Aur, bcs,
    solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.000001,"absolute_tolerance":tol_abs,"maximum_iterations":200}})
 
 A = Aur.sub(0, deepcopy=True)
 u = Aur.sub(1, deepcopy=True)
 r = Aur.sub(2, deepcopy=True)
 
 #Saving the output in a .xdmf file.
 Aur_split = Aur.split(True)
 Aur_out = XDMFFile('test-0-ConstraintLoopLength.xdmf')
 Aur_out.write_checkpoint(Aur_split[0], "A", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
 Aur_out.close()
 Aur_out = XDMFFile('test-1-ConstraintLoopLength.xdmf')
 Aur_out.write_checkpoint(Aur_split[1], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
 Aur_out.close()
 with open("test-2-ConstraintLoopLength.txt", "w") as file:
     print(float(Aur_split[2]), file=file)
 
 
 #Printing energy density and constraint violation.
 pie = assemble((1/(Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
 print("Energy density =", pie)
 Constraint = assemble( (u-0.5)*dx)
 print("Constraint violated by =", Constraint)
 print("length = ", float(Lx))
 print("discretization #=",int(Lx)*10)
 
 #Plotting
 if ii%int(nn) == 0:
  plot(u)
  plt.title(r"$u(x)$",fontsize=26)
  plt.show()
  plot(A)
  plt.title(r"$A(x)e_2$",fontsize=26)
  plt.show()

