#Here we solve the 1D Ginzbug Landau problem with an applied magnetic field.
#the whole formulation is presented in OneNote UH/superConductivity/Coding/1D Ginzburg Landau fenics.
#This is a modification of the original code in that we have added an integral constraint
#∫(u-1/2)=0.
#look specifically at sec-V
#The lagrange multiplier is r. 
#======================================================================================================
#The way the Code works
#1. The input to the code is:
#   a. The external field
#      While it should be 1/sqrt(2), we allow this as i/p so that we cna fine tune it and find H_c for 
#      a finite domain with boundary effects.
#   b. The relaxation parameter
#   c. The absolute tolerance
#2. When reading from and writing into respective files,
#   we are writing the lagrange multiplier as a constant function
#   When reading the functions, we interpolate onto a space VAu.
#======================================================================================================
#The code works perfectly.


from dolfin import *
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt


#Create mesh and define function space
Lx=10
mesh = fe.IntervalMesh(100,0,Lx)
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


#Dirichlet BC for left bdry
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
#Aur = interpolate( Expression(("1","0.0", "1.5"), degree=2), V)#SC phase as initial cond.
#Aur = interpolate( Expression(("H*x[0]","0", "0"), H=H, degree=2), V)#Normal phase as initial condiiton
##Coexistence of phase as initial condition
#ul = Expression('0', degree=2, domain=mesh)
#Al = Expression('H*(0.5*Lx-x[0])', H=H, Lx=Lx, degree=2, domain=mesh)
#ur = Expression('1', degree=2, domain=mesh)
#Ar = Expression('0', degree=2, domain=mesh)
#Aur = interpolate( Expression(('x[0] <= 0.5*Lx + DOLFIN_EPS ? Al : Ar', 'x[0]<=0.5*Lx+DOLFIN_EPS ? ul : ur', '11'), ul=ul, ur=ur, Al=Al, Ar=Ar, Lx=Lx, degree=2), V)
#For 1D Vortex Solution.
#Aur = interpolate( Expression(("sqrt(2*tanh(x[0]+0.89)*tanh(x[0]+0.89)-1)","-sqrt(2)*sqrt(1-tanh(x[0]+0.89)*tanh(x[0]+0.89))","1"), degree=3), V)#1D vortex solution.
#---------------------------------------------------------------------------------------------------------------
#Reading input from a .xdmf file.
Aur = Function(V)
A = Function(Vcoord)
u = Function(Vcoord)
r = Function(RFnSp)
data = np.loadtxt('Cr-1.txt')
y0 = data
r = interpolate(Constant(float(y0)),RFnSp)
A_in =  XDMFFile("CA-1.xdmf")
A_in.read_checkpoint(A,"A",0)
u_in =  XDMFFile("Cu-1.xdmf")
u_in.read_checkpoint(u,"u",0)
assign(Aur,[A,u,r])

(A, u, r) = split(Aur)

#Scaling down initial residue for large domains.


F = (-(1-u**2)*u*du + (1/kappa**2)*inner(grad(u), grad(du)) + A**2*u*du + 0.5*r*du + dr*(u-0.5) + u**2*A*dA + inner(grad(A), grad(dA)))*dx + H*dA*ds
solve(F == 0, Aur, bcs,
   solver_parameters={"newton_solver":{"convergence_criterion":"residual","relaxation_parameter":rlx_par,"relative_tolerance":0.001,"absolute_tolerance":tol_abs,"maximum_iterations":4000}})

A = Aur.sub(0, deepcopy=True)
u = Aur.sub(1, deepcopy=True)
r = Aur.sub(2, deepcopy=True)

##Save solution in a .xdmf file
Aur_split = Aur.split(True)
Aur_out = XDMFFile('CA-4.xdmf')
Aur_out.write_checkpoint(Aur_split[0], "A", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Aur_out.close()
Aur_out = XDMFFile('Cu-4.xdmf')
Aur_out.write_checkpoint(Aur_split[1], "u", 0, XDMFFile.Encoding.HDF5, False) #false means not appending to file
Aur_out.close()
with open("Cr-4.txt", "w") as file:
    print(float(Aur_split[2]), file=file)


pie = assemble((1/(Lx))*((1-u**2)**2/2 + (1/kappa**2)*inner(grad(u), grad(u)) + A**2*u**2 + inner(grad(A-Ae), grad(A-Ae)))*dx )
print("Energy density =", pie)
Constraint = assemble( (u-0.5)*dx)
print("Constraint violated by =", Constraint)


plot(u)
plt.title(r"$u(x)$ for domain [0,"+str(Lx)+"] for H= "+str(Hin)+", with rlx_par "+str(rlx_par_in)+" and abs_tol "+str(tol_abs_in),fontsize=26)
plt.show()
plot(A)
plt.title(r"$A(x)$ for domain [0,"+str(Lx)+"] for H= "+str(Hin)+", with rlx_par "+str(rlx_par_in)+" and abs_tol "+str(tol_abs_in),fontsize=26)
plt.show()

