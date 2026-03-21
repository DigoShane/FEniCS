#General code to solve 1D thermoelastic problems
#div(k\nabla T)=0
#T(0)=50
#T(1)=T0
#\sigma=Eu'-\alpha*E*Delta T
#where Delta T= T-T0, with T0 being the initial temp (unif 0)
#(\sigma)'+f(x)=0
#u(0)=0
#u(1)=0
from fenics import *

#################################################################
#	Defining the basic variables
#################################################################
L = 1 #length of domian
mesh = IntervalMesh(32,0,1)
# Define Dirichlet boundary (x = 0)
tol = 1e-14
def LboundaryT(x, on_boundary):
    return on_boundary and abs(x[0]) < tol
def RboundaryT(x, on_boundary):
    return on_boundary and near(x[0],1,tol)
def Dboundaryu(x, on_boundary):
    return on_boundary and near(x[0],1,tol) or near(x[0],0,tol) 
K = Constant(1)
EA = Constant(1)
alpha = Constant(0.2)
f = Constant(0)

#################################################################
#	Solving the Heat transfer problem first
#################################################################
VT = FunctionSpace(mesh, "CG", 3)
dT = TestFunction(VT)
T = TrialFunction(VT)
Delta_T = Function(VT, name="Temp increase")

aT = dot(K*grad(T),grad(dT))*dx
LT = Constant(0)*dT*dx

## Define boundary condition
T0 = Constant(0.0)
bcT = [DirichletBC(VT, T0, RboundaryT),
       DirichletBC(VT, Constant(50), LboundaryT)]
solve(aT == LT, Delta_T, bcT)

#################################################################
#	Solving the Mechanical coupled problem
#################################################################
##,def eps(v):
##, return v.dx[0]
##,# return grad(v)
##,def sigma(v,dT):
##, return eps(v)-dT
##,# return EA*eps(v)-alpha*EA*dT

Vu = FunctionSpace(mesh, "CG", 3)
QT = FunctionSpace( mesh, "CG", 2)
u = TrialFunction(Vu)
v = TestFunction(Vu)
fT = project( Delta_T.dx(0), QT) #Thermal load

#Wint = inner(sigma(u,Delta_T),eps(du))*dx
au = inner(EA*grad(u), grad(v))*dx
Lu = inner(alpha*EA*fT,v)*dx + inner(f, v)*dx

## Define boundary condition
u0 = Constant(0.0)
bcu = DirichletBC(Vu, u0, Dboundaryu)

# Compute solution
u = Function(Vu, name="Displacement")
solve(au == Lu, u, bcu)

# Save solution in VTK format
file = File("1DT.pvd")
file << Delta_T

# Save solution in VTK format
file1 = File("1Du.pvd")
file1 << u
