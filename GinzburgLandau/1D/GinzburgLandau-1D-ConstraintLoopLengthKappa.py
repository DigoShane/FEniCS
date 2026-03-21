#Here we solve the 1D Ginzbug Landau problem with an applied critical field, H=1/√2.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/Ginsburg landau fenicsi/sec-V.
#This is a modification of the original code in that we have added an integral constraint.
#------------------------------------------------------------------------------------------------------
#HOW IS THIS CODE DIFFERENT:-
#1. The way this code runs is that there are two loops, an outer loop over kappa, and an inner loop over
#   the length. The outer loop is run for L=10, the output of the outer loop is used as input for the 
#   inner loop.



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
H = Constant('0.7071');#Constant(Hin);
nn = 100;#No. of plots to skip over
Lmax = 501;
Kmax = 10.1;
Kmin = 1.3;

for jj in np.arange(Kmin, Kmax, 0.1):
  kappa = Constant(jj);

  #Loop over other elements
  for ii in np.arange(10, Lmax,1):
   Lx = Constant(float(ii));
   mesh = fe.IntervalMesh(np.ceil(Lx*10/kappa),0,Lx)
   x = SpatialCoordinate(mesh)
   VA = FiniteElement("CG", mesh.ufl_cell(), 2)
   Vu = FiniteElement("CG", mesh.ufl_cell(), 2)
   R = FiniteElement("Real", mesh.ufl_cell(), 0)
   V = FunctionSpace(mesh, MixedElement(VA, Vu, R))
   RFnSp = FunctionSpace(mesh, "Real", 0)
   Vcoord = FunctionSpace(mesh, "Lagrange", 2)#This is for ExtFile
   Ae = H*x[0]
   
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
  
   if ii == 10:
    #Coexistence of phase as initial condition
    ul = Expression('0', degree=2, domain=mesh)
    ur = Expression('1', degree=2, domain=mesh)
    Ar = Expression('0', degree=2, domain=mesh)
    Al = Expression('0.7071*(x[0]-0.5*L)', L=ii, degree=2, domain=mesh)
    Aur = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur', '11'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
   else:
    #Reading input from a .xdmf file.
    Aur = Function(V)
    A = Function(Vcoord)
    u = Function(Vcoord)
    r = Function(RFnSp)
    data = np.loadtxt('test-2-ConstraintLoopLengthKappa.txt')
    y0 = data
    r = interpolate(Constant(float(y0)),RFnSp)
    A_in =  XDMFFile("test-0-ConstraintLoopLengthKappa.xdmf")
    A_in.read_checkpoint(A,"A",0)
    u_in =  XDMFFile("test-1-ConstraintLoopLengthKappa.xdmf")
    u_in.read_checkpoint(u,"u",0)
    assign(Aur,[A,u,r])
 
   (A, u, r) = split(Aur)
     
   F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + 0.5*r*du + dr*(u-0.5) + u**2*A*dA + inner(grad(A), grad(dA)))*dx + H*dA*ds#testing with original problem.
   
   if ii==10:
     rlx=float(0.001)
   else:
     rlx=float(1)
   
   solve(F == 0, Aur, bcs,
      solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx,"relative_tolerance":0.001,"absolute_tolerance":0.000001,"maximum_iterations":100000}})

   A = Aur.sub(0, deepcopy=True)
   u = Aur.sub(1, deepcopy=True)
   r = Aur.sub(2, deepcopy=True)
   
   #Saving the output in a .xdmf file.
   Aur_split = Aur.split(True)
   Aur_out = XDMFFile('test-0-ConstraintLoopLengthKappa.xdmf')
   Aur_out.write_checkpoint(Aur_split[0], "A", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
   Aur_out.close()
   Aur_out = XDMFFile('test-1-ConstraintLoopLengthKappa.xdmf')
   Aur_out.write_checkpoint(Aur_split[1], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
   Aur_out.close()
   with open("test-2-ConstraintLoopLengthKappa.txt", "w") as file:
       print(float(Aur_split[2]), file=file)
   
   
   #Printing energy density and constraint violation.
   pie = assemble((1/(Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
   print("Energy density =", pie)
   Constraint = assemble( (u-0.5)*dx)
   print("Constraint violated by =", Constraint)
   print("length = ", float(Lx))
   
   #Plotting
   if (int(ii)%int(nn) == 0) or (int(ii)==10):
    plot(u)
    plt.title(r"$u$,$L$="+str(ii)+",$\kappa$="+str(jj)+"E="+str(round(pie,6)),fontsize=26)
    #plt.show()
    plt.savefig('u,L=,'+str(ii)+',kappa='+str(jj) + '.png')
    plt.close()
    plot(A)
    plt.title(r"$A,L=$"+str(ii)+",$\kappa$="+str(jj)+"E="+str(round(pie,6)),fontsize=26)
    #plt.show()
    plt.savefig('A,L=,'+str(ii)+',kappa='+str(jj) + '.png')
    plt.close()



