#General code to solve 1D Ginzburg Landau functional
#\int_{\Omega}\varphi(u)-b.u dV+\int_{\Gamma_D}T.u dS
#subject to u(\Gamma_D) specified
#where \Omega=[0,1]^3
#\Gamma_{D0} = 0x(0,1)^2
#\Gamma_{D1} = 1x(0,1)^2
#\Gamma_N = \partial\Omega \backslash \Gamma_D
#u\vert_{\Gamma_{D0}} = (0.5 + (y-0.5)\cos(\pi/3) - (z-0.5)\sin(\pi/3)-y)/2 e_2 + (0.5 + (y-0.5)\sin(\pi/3) - (z-0.5)\cos(\pi/3)-x)/2 e_3
#u\vert_{\Gamma_{D1}} = 0
#T=0.1 e_1 on \Gamma_N
#B = -0.5 e_2
#E = 10.0
#\nu = 0.3

from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
ffc_options = {"optimize": True, \
               "eliminate_zeros": True, \
               "precompute_basis_const": True, \
               "precompute_ip_const": True}

# Create mesh and define function space
#!!xDx!!mesh = UnitCubeMesh(2, 1, 1)
domain_vertices = [Point(0.0, 0.0, 0.0),
                   Point(1.0, 0.0, 0.0),
                   Point(0.0, 1.0, 0.0),
                   Point(0.0, 0.0, 1.0),
                   Point(1.0, 1.0, 0.0),
                   Point(0.0, 1.0, 1.0),
                   Point(1.0, 0.0, 1.0),
                   Point(1.0, 1.0, 1.0)]
PolygonalMeshGenerator.generate(mesh, domain_vertices, 0.25)
V = VectorFunctionSpace(mesh, "Lagrange", 1)

#!!xDx!!# Mark boundary subdomians
#!!xDx!!left =  CompiledSubDomain("near(x[0], side) && on_boundary", side = 0.0)
#!!xDx!!right = CompiledSubDomain("near(x[0], side) && on_boundary", side = 1.0)
#!!xDx!!
#!!xDx!!# Define Dirichlet boundary (x = 0 or x = 1)
#!!xDx!!c = Expression(("0.0", "0.0", "0.0"))
#!!xDx!!r = Expression(("scale*0.0",
#!!xDx!!                "scale*(y0 + (x[1] - y0)*cos(theta) - (x[2] - z0)*sin(theta) - x[1])",
#!!xDx!!                "scale*(z0 + (x[1] - y0)*sin(theta) + (x[2] - z0)*cos(theta) - x[2])"),
#!!xDx!!                scale = 0.5, y0 = 0.5, z0 = 0.5, theta = pi/3)                                               
#!!xDx!!
#!!xDx!!bcl = DirichletBC(V, c, left)
#!!xDx!!bcr = DirichletBC(V, r, right)
#!!xDx!!bcs = [bcl, bcr]
#!!xDx!!
#!!xDx!!# Define functions
#!!xDx!!du = TrialFunction(V)            # Incremental displacement
#!!xDx!!v  = TestFunction(V)             # Test function
#!!xDx!!u  = Function(V)                 # Displacement from previous iteration
#!!xDx!!B  = Constant((0.0, -0.5, 0.0))  # Body force per unit volume
#!!xDx!!T  = Constant((0.1,  0.0, 0.0))  # Traction force on the boundary
#!!xDx!!
#!!xDx!!# Kinematics
#!!xDx!!d = u.geometric_dimension()
#!!xDx!!I = Identity(d)             # Identity tensor
#!!xDx!!F = I + grad(u)             # Deformation gradient
#!!xDx!!C = F.T*F                   # Right Cauchy-Green tensor
#!!xDx!!
#!!xDx!!# Invariants of deformation tensors
#!!xDx!!Ic = tr(C)
#!!xDx!!J  = det(F)
#!!xDx!!
#!!xDx!!# Elasticity parameters
#!!xDx!!E, nu = 10.0, 0.3
#!!xDx!!mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
#!!xDx!!
#!!xDx!!# Stored strain energy density (compressible neo-Hookean model)
#!!xDx!!psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
#!!xDx!!
#!!xDx!!# Total potential energy
#!!xDx!!Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds
#!!xDx!!
#!!xDx!!# Elasticity parameters
#!!xDx!!E, nu = 10.0, 0.3
#!!xDx!!mu, lmbda = Constant(E/(2*(1 + nu))), Constant(E*nu/((1 + nu)*(1 - 2*nu)))
#!!xDx!!
#!!xDx!!# Stored strain energy density (compressible neo-Hookean model)
#!!xDx!!psi = (mu/2)*(Ic - 3) - mu*ln(J) + (lmbda/2)*(ln(J))**2
#!!xDx!!
#!!xDx!!# Total potential energy
#!!xDx!!Pi = psi*dx - dot(B, u)*dx - dot(T, u)*ds
#!!xDx!!
#!!xDx!!# Compute first variation of Pi (directional derivative about u in the direction of v)
#!!xDx!!F = derivative(Pi, u, v)
#!!xDx!!
#!!xDx!!# Compute Jacobian of F
#!!xDx!!J = derivative(F, u, du)
#!!xDx!!
#!!xDx!!# Compute solution
#!!xDx!!solve(F == 0, u, bcs, J=J,
#!!xDx!!      form_compiler_parameters=ffc_options)
#!!xDx!!
#!!xDx!!# Save solution in VTK format
#!!xDx!!file = File("Displacement.pvd")
#!!xDx!!file << u
#!!xDx!!
#!!xDx!!# Plot and hold solution
#!!xDx!!plot(u, mode = "displacement", interactive = True)
