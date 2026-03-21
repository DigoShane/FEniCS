#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#look specifically at sec-IV
#The domain for the problem is [0,Lx] for initial conditions for "bulk SC", "bulk Normal", "Coexistence of Phase"
#We only consider the half domain for the initial conditions "1D Vortex"
#The math for which is presented in "UH One Note/superConductivity/1D domaim walls/High kappa limit in 1D."
#We have presented the closed form there.
#=====================================================================================================
#ISSUES WITH THE CODE:-
#1.Numerical Tests and corresponding hypothesis 
#   I did some numerical tests and came up with a hypothesis on whats going on with the code. 
#   To see this, check out Overleaf shoham.sen16/Things to Discuss/Sec. 17/12/23 P+L/Text. Things Discussed.
#   The above explains the numerical tests and the corresponding hypothesis.
#   This suggests that we need to work on the constrained code which might yield a better result.
#
#Ans. One thing that was bothering us is that the code would give really large residues and would get stuck in large loops.
#     Based on the analysis by Liping, it seemed that every iteration should return a better guess with the residue decreasing.
#     Turns out the large residue was due to the large domain size, so we can scale the F=0 condition by F/c=0 to get a better
#     residue. I was able to implement the relaxation parameter which gave better results. However taking large no. of iterations.
#     To get around this, I found a way to write the output to a text file and reuse it as input for the subsequent runs. 
#     Eventually converging to a minimum.
#-----------------------------------------------------------------------------------------------------
#2.Analyzing the Newton Rhapson Method
#  The analysis of the Newton Rhapson method is presented in
#  One Note. UH/superConductivity/Coding/Analyzing the Newton rhapson Method.
#=====================================================================================================

from dolfin import *
import fenics as fe
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


# Optimization options for the form compiler
#parameters["form_compiler"]["cpp_optimize"] = True
parameters["krylov_solver"]["nonzero_initial_guess"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}


#Create mesh and define function space
Lx=500
pord=2
mesh = fe.IntervalMesh(5000,0,Lx)
x = SpatialCoordinate(mesh)
VA = FiniteElement("CG", mesh.ufl_cell(), pord)
Vu = FiniteElement("CG", mesh.ufl_cell(), pord)
Vcoord = FunctionSpace(mesh, "Lagrange", pord)#We use this for read & write using ExtFile.py
V = FunctionSpace(mesh, MixedElement(VA, Vu))

# Define functions
dAu = TrialFunction(V)
(dA, du) = split(dAu)


#Dirichlet BC for left bdry
def boundary_L(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], 0, tol)
def boundary_R(x, on_boundary):
    tol = 1E-24
    return on_boundary and near(x[0], Lx, tol)

##For the 1D vortex solution
#bc1 = DirichletBC(V.sub(1), 1, boundary_L)
#bc2 = DirichletBC(V.sub(1), 1, boundary_R)
#bc3 = DirichletBC(V.sub(0), 0, boundary_R)
#bc4 = DirichletBC(V.sub(0), 0, boundary_L)
#bcs = [bc1, bc2, bc3, bc4];

#For other cases
bc1 = DirichletBC(V.sub(1), 0, boundary_L)
bc2 = DirichletBC(V.sub(1), 1, boundary_R)
bc3 = DirichletBC(V.sub(0), 0, boundary_R)
bcs = [bc1, bc2, bc3];

# Parameters
kappa = Constant(1);
Hin = input("External Magnetic field? ")
H = Constant(Hin);
rlx_par_in = input("relaxation parameter? ")
rlx_par = Constant(rlx_par_in);
tol_abs_in = input("absolute tolerance? ")
tol_abs = Constant(tol_abs_in);
Ae = H*x[0]

#-----------------------------------------------------------------------------------------------------------------
#!!xDx!! ##!!xDx!! Newton rhapson Approach
#-----------------------------------------------------------------------------------------------------------------
#Compute first variation of Pi (directional derivative about u in the direction of v)
##Other cases
#Au = interpolate( Expression(("0","1.5"), degree=1), V)#SC phase as initial cond.
#Au = interpolate( Expression(("H*x[0]","0"), H=H, degree=2), V)#Normal phase as initial condiiton
#Au = interpolate( Expression(("0.5*h*x[0]","0.5*tanh(x[0]-0.5*l)+0.5"), l=Lx, h=H,  degree=pord), V)#coexistence of phase as initial cond.
#Coexistence of phase as initial condition
ul = Expression('0', degree=1, domain=mesh)
Al = Expression('H*x[0]', H=H, degree=1, domain=mesh)
ur = Expression('1', degree=1, domain=mesh)
Ar = Expression('0', degree=1, domain=mesh)
Au = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=1), V)
#----------------------------------------------------------------------------------------------------------------
##1D Vortex solution
#psir = Expression('sqrt(2*tanh(x[0]-0.5*Lx+0.89)*tanh(x[0]-0.5*Lx+0.89)-1)', Lx=Lx, degree=2, domain=mesh)
#Ar = Expression('-sqrt(2)*sqrt(1-tanh(x[0]-0.5*Lx+0.89)*tanh(x[0]-0.5*Lx+0.89))', Lx=Lx, degree=2, domain=mesh)
#psil = Expression('sqrt(2*tanh(x[0]-0.5*Lx-0.89)*tanh(x[0]-0.5*Lx-0.89)-1)', Lx=Lx, degree=2, domain=mesh)
#Al = Expression('-sqrt(2)*sqrt(1-tanh(x[0]-0.5*Lx-0.89)*tanh(x[0]-0.5*Lx-0.89))', Lx=Lx, degree=2, domain=mesh)
#Au = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? psil : psir'), psil=psil, psir=psir, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)#1D vortex solution.
#---------------------------------------------------------------------------------------------------------------
##Reading input from a .xdmf file.
#Au = Function(V)
#A = Function(Vcoord)
#u = Function(Vcoord)
#A_in =  XDMFFile("A-10.xdmf")
#A_in.read_checkpoint(A,"A",0)
#u_in =  XDMFFile("u-10.xdmf")
#u_in.read_checkpoint(u,"u",0)
#assign(Au,[A,u])


(A, u) = split(Au)


Scl=1;

##For 1D Vortex Solutions.
#F = Scl*(-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + u**2*A*dA + inner(grad(A), grad(dA)))*dx
#Other cases
F = Scl*(-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + u**2*A*dA + inner(grad(A), grad(dA)))*dx +Scl*H*dA*ds
#solver.parameters.nonzero_initial_guess = True
solve(F == 0, Au, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.001,"absolute_tolerance":tol_abs,"maximum_iterations":100}})


A = Au.sub(0, deepcopy=True)
u = Au.sub(1, deepcopy=True)

Au_split = Au.split(True)

#Save solution in a .xdmf file
Au_out = XDMFFile('A-10.xdmf')
Au_out.write_checkpoint(Au_split[0], "A", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Au_out.close()
Au_out = XDMFFile('u-10.xdmf')
Au_out.write_checkpoint(Au_split[1], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Au_out.close()


pie = assemble((1/(Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
print("Energy density is",pie)

fig=plot(u)
plt.title(r"$u(x)$",fontsize=26)
plt.show()
fig=plot(A)
plt.title(r"$A(x)e_2$",fontsize=26)
plt.show()


