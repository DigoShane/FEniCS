from dolfin import *
from mshr import *
#import numpy as np

L, H = 5, 0.3
mesh = RectangleMesh(Point(0., 0.), Point(L, H), 100, 10, "crossed")

def lateral_sides(x, on_boundary):
    return (near(x[0], 0) or near(x[0], L)) and on_boundary
def bottom(x, on_boundary):
    return near(x[1], 0) and on_boundary
def top(x, on_boundary):
    return near(x[1], H) and on_boundary

VT = FunctionSpace(mesh, "CG", 1)
T_, dT = TestFunction(VT), TrialFunction(VT)
Delta_T = Function(VT, name="Temperature increase")
T1 = Function(VT, name="Initial Temperature")
T2 = Function(VT, name="Final Temperature")
aT = dot(grad(dT), grad(T_))*dx
LT = Constant(0)*T_*dx

#Solving for the initial Temp distribution
bcT = [DirichletBC(VT, Constant(0.), bottom),
       DirichletBC(VT, Constant(0.), top),
       DirichletBC(VT, Constant(10.), lateral_sides)]
solve(aT == LT, T1, bcT)

#Solving for the final Temp distribution
bcT = [DirichletBC(VT, Constant(50.), bottom),
       DirichletBC(VT, Constant(0.), top),
       DirichletBC(VT, Constant(10.), lateral_sides)]
solve(aT == LT, T2, bcT)

Delta_T = project(T2-T1, VT)

# Save solution in VTK format
vtkfile = File("ThermoElasticFigures/2DTemp1.pvd")
vtkfile << T1
vtkfile = File("ThermoElasticFigures/2DTemp2.pvd")
vtkfile << T2
vtkfile = File("ThermoElasticFigures/2D_DeltaTemp.pvd")
vtkfile << Delta_T

E = Constant(50e3)
nu = Constant(0.2)
mu = E/2/(1+nu)
lmbda = E*nu/(1+nu)/(1-2*nu)
alpha = Constant(1e-5)

f = Constant((0, 0))

def eps(v):
    return sym(grad(v))
def sigma(v, dT):
    return (lmbda*tr(eps(v))- alpha*(3*lmbda+2*mu)*dT)*Identity(2) + 2.0*mu*eps(v)

Vu = VectorFunctionSpace(mesh, 'CG', 2)
du = TrialFunction(Vu)
u_ = TestFunction(Vu)
Wint = inner(sigma(du, Delta_T), eps(u_))*dx
aM = lhs(Wint)
LM = rhs(Wint) + inner(f, u_)*dx

bcu = DirichletBC(Vu, Constant((0., 0.)), lateral_sides)

u = Function(Vu, name="Displacement")

solve(aM == LM, u, bcu)

# Save solution in VTK format
FE = FiniteElement('Lagrange', mesh.ufl_cell(),3)
V = FunctionSpace(mesh, FE)
U2 = Function(V)
w0 = Function(V)
U2 =u[1]
w0 = project(U2, V)
vtkfile = File('ThermoElasticFigures/2Ddisp_NoForce.pvd')
vtkfile << w0

rho_g = 2400*9.81e-6
f.assign(Constant((0., -rho_g)))
solve(aM == LM, u, bcu)

# Save solution in VTK format
FE = FiniteElement('Lagrange', mesh.ufl_cell(),3)
V = FunctionSpace(mesh, FE)
U2 = Function(V)
w0 = Function(V)
U2 =u[1]
w0 = project(U2, V)
vtkfile = File('ThermoElasticFigures/2Ddisp_YesForce.pvd')
vtkfile << w0
