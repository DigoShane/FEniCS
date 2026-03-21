from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True #C++ compiler optimizations when compiling the generated code
parameters["form_compiler"]["representation"] = "uflacs"

# Create mesh and define function space
mesh = UnitSquareMesh(24, 16)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

# Mark boundary subdomians
class Left(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[0],0) and on_boundary
class Right(SubDomain):
 def inside(self, x, on_boundary):
  return near(x[0],1) and on_boundary
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
Left().mark(facets, 2)
ds = Measure('ds', subdomain_data=facets)

# Define Dirichlet boundary (x = 0 or x = 1)
c = Constant((0.0, 0.0))
r = Expression(("scale*0.0",
                "scale*(y0 + (x[1] - y0)*sin(theta))"),
                scale = 0.5, y0 = 0.5, theta = pi/3, degree=2)

bcs = [DirichletBC(V, r, facets, 0), DirichletBC(V, c, facets, 2)]

# Define functions
du = TrialFunction(V)            # Incremental displacement
v  = TestFunction(V)             # Test function
u  = Function(V)                 # Displacement from previous iteration
B  = Constant((0.0, -0.5))  # Body force per unit volume
T  = Constant((0.1,  0.0))  # Traction force on the boundary

# Kinematics
d = len(u)		    # dimension
I = Identity(d)             # Identity tensor
F = I + grad(u)             # Deformation gradient
C = F.T*F                   # Right Cauchy-Green tensor

# Invariants of deformation tensors
Ic = tr(C)
J  = det(F)

# Elasticity parameters
E, nu = 10.0, 0.3
mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))

# Stored strain energy density (compressible neo-Hookean model)
psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2

# Total potential energy
Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds

# Compute first variation of Pi (directional derivative about u in the direction of v)
F1 = derivative(Pi, u, v)

# Compute Jacobian of F
J1 = derivative(F1, u, du)

# Solve variational problem
solve(F1 == 0, u, bcs, J=J1)

# Save solution in VTK format
FE = FiniteElement('Lagrange', mesh.ufl_cell(),3)
V = FunctionSpace(mesh, FE)
U1 = Function(V)
w0 = Function(V)
U1 =u[1]
w0 = project(U1, V)
vtkfile = File("Non-LinearFigure/displacement.pvd")
vtkfile << w0

