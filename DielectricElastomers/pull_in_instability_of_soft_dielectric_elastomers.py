import dolfin
from dolfin import *
import numpy as np
#import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt

import os
import shutil

parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["quadrature_degree"] = 4
length = 1.

N = input("Enter the number of elements in the x,y-direction: ")
N = int(N)

#Mesh
mesh = RectangleMesh(Point(0.,0.),Point(length,length), N, N, "crossed")
x = SpatialCoordinate(mesh)

#Pick up on the boundary entities of the created mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],length) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],length) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0.0) and on_boundary

# Mark boundary subdomains
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
DomainBoundary().mark(facets, 5)  # First, mark all boundaries with common index
# Next mark specific boundaries
Left().mark(facets,  1)
Bottom().mark(facets,2)
Right().mark(facets, 3)
Top().mark(facets, 4)

# Define the boundary integration measure "ds".
ds = Measure('ds', domain=mesh, subdomain_data=facets)

# Mechanical parameters
Geq_0   = 15         # Shear modulus, kPa
Kbulk   = 1e3*Geq_0  # Bulk modulus, kPa
I_m     = 175        # Gent locking paramter
# Electrostatic  parameters
vareps_0 = Constant(8.85E-3)         #  permittivity of free space pF/mm
vareps_r = Constant(5)             #  relative permittivity, dimensionless
vareps   = vareps_0*vareps_r         #  permittivity of the material

# Applied potential on the top electrode (physical units, consistent with vareps)
phi_app = float(input("Enter the applied potential on the top electrode: "))

# Fixed spatial distribution of potential on the top boundary
phiTop = Expression("phi_app*sin(pi*x[0]/L)",
                    phi_app=phi_app, L=length, pi=np.pi, degree=3)

# Define function space, both vectorial and scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 2) # For displacement
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1) # For pressure and electric potential
#
TH = MixedElement([U2, P1, P1])   
ME = FunctionSpace(mesh, TH)      

# Define actual functions with the required DOFs
w = Function(ME)
u, p, phi = split(w) # dispalacement u, pressure p, potential, phi

# soln at prev step.
w_old = Function(ME)
u_old, p_old, phi_old = split(w_old)   # old values

# Test functions
w_test = TestFunction(ME)   # Test function
u_test, p_test, phi_test = split(w_test)  # test fields

#Trial functions needed for automatic differentiation
dw = TrialFunction(ME)

#  They are also used  later for visualization of results
W2 = FunctionSpace(mesh, U2)  # Vector space
W  = FunctionSpace(mesh, P1)   # Scalar space


# Output Folder Setup:
base_png_dir = "png_results"

total_stress_png_dir = os.path.join(base_png_dir, "total_cauchy_stress")
maxw_stress_png_dir  = os.path.join(base_png_dir, "maxwell_cauchy_stress")
cauchy_stress_png_dir = os.path.join(base_png_dir, "cauchy_stress")
Eref_png_dir         = os.path.join(base_png_dir, "electric_field_reference")
top_surface_stretch_png_dir = os.path.join(base_png_dir, "top_surface_stretch")

center_node_values_dir = os.path.join(base_png_dir, "center_node_values")

displacement_png_dir = os.path.join(base_png_dir, "displacement")

all_png_dirs = [ total_stress_png_dir, maxw_stress_png_dir, 
                 cauchy_stress_png_dir, Eref_png_dir,
                 top_surface_stretch_png_dir,
                 displacement_png_dir, center_node_values_dir]

# Clear old PNG results only
if os.path.exists(base_png_dir):
    shutil.rmtree(base_png_dir)

# Create output folders
for folder in all_png_dirs:
    os.makedirs(folder, exist_ok=True)

print("Created PNG output folders:")
for folder in all_png_dirs:
    print("  ", os.path.abspath(folder))

# Gradient of vector field u
def pe_grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, 0]])

# Gradient of scalar field y
# (just need an extra zero for dimensions to work out)
def pe_grad_scalar(y):
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])

# Plane strain deformation gradient
def F_pe_calc(u):
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F = Id + grad(u)            # 2D Deformation gradient
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, 1]]) # Full pe F

# Generalized shear modulus for Gent model
def Geq_Gent_calc(u):
    F = F_pe_calc(u)
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = tr(Cdis)
    z = I1-3
    z   = conditional( gt(z, I_m), 0.95*I_m, z ) # Keep from blowing up
    Geq_Gent  = Geq_0 * (I_m/(I_m - z))
    return Geq_Gent

# Mechanical Cauchy stress for Gent material
def T_mech_calc(u,p):
    Id = Identity(3)
    F   = F_pe_calc(u)
    J = det(F)
    B = F*F.T
    Bdis = J**(-2/3)*B
    Geq  = Geq_Gent_calc(u)
    T_mech = (1/J)* Geq * dev(Bdis) - p * Id
    return T_mech

# Maxwell contribution to the Cauchy stress
def T_maxw_calc(u,phi):
    F = F_pe_calc(u)
    e_R  = - pe_grad_scalar(phi)    # referential electric field
    e_sp = inv(F.T)*e_R   # spatial electric field
    # Spatial Maxwel stress
    T_maxw = vareps*(outer(e_sp,e_sp) - 1/2*(inner(e_sp,e_sp))*Identity(3))
    return T_maxw

def T_mat_calc(u, p, phi):
    Id = Identity(3)
    F   = F_pe_calc(u)
    J = det(F)
    T_mech = T_mech_calc(u,p)
    T_maxw = T_maxw_calc(u,phi)
    T      = T_mech + T_maxw
    Tmat   = J * T * inv(F.T)
    return Tmat

def Dmat_calc(u, phi):
    F = F_pe_calc(u)
    J = det(F)
    C = F.T*F
    e_R  = - pe_grad_scalar(phi) # reference electric field
    Dmat = vareps*J*inv(C)*e_R
    return Dmat

# Some kinematical quantities
F =  F_pe_calc(u)
J = det(F)
C =  F.T*F
Fdis = J**(-1/3)*F
Cdis = J**(-2/3)*C

# Mechanical Cauchy stress
T_mech = T_mech_calc(u, p)
# Electrostatic Cauchy stress
T_maxw =T_maxw_calc(u, phi)
# Piola stress
Tmat = T_mat_calc(u, p, phi)
# Referential electric displacement
Dmat = Dmat_calc(u, phi)

# The weak form for the equilibrium equation
Res_0 =  inner(Tmat, pe_grad_vector(u_test) )*dx
# The weak form for the pressure
Res_1 =  dot((p/Kbulk + ln(J)/J) , p_test)*dx
#  The weak form for Gauss's equation
Res_2 = inner(Dmat, pe_grad_scalar(phi_test))*dx

# Total weak form
Res  =  Res_0 + Res_1 + Res_2

# Automatic differentiation tangent:
a = derivative(Res, w, dw)

def save_scalar_png(scalar_function, folder, name, phi_app):
    os.makedirs(folder, exist_ok=True)

    coords = mesh.coordinates()
    cells = mesh.cells()
    values = scalar_function.compute_vertex_values(mesh)

    plt.figure(figsize=(6, 4))
    p_plot = plt.tripcolor( coords[:, 0], coords[:, 1], values, triangles=cells, shading="gouraud")
    plt.colorbar(p_plot)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()

    filename = f"{name}_phi_{phi_app:.4f}.png"
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def project_and_save_scalar(expr, W, name, phi_app, folder):
    field = project(expr, W)
    field.rename(name, " ")
    save_scalar_png(field, folder, name, phi_app)

    return field

def save_colored_vector_arrows(expr_x, expr_y, W, folder, name, phi_app,
                            max_arrows_per_direction=20, normalize_arrows=True):
    os.makedirs(folder, exist_ok=True)

    Ex_fun = project(expr_x, W)
    Ey_fun = project(expr_y, W)
    coords = mesh.coordinates()
    Ex = Ex_fun.compute_vertex_values(mesh)
    Ey = Ey_fun.compute_vertex_values(mesh)

    Emag = np.sqrt(Ex**2 + Ey**2)

    num_vertices = coords.shape[0]
    target_arrows = max_arrows_per_direction**2
    stride = max(1, int(num_vertices / target_arrows))

    arrow_coords = coords[::stride]
    arrow_Ex = Ex[::stride]
    arrow_Ey = Ey[::stride]
    arrow_Emag = Emag[::stride]

    if normalize_arrows: #default val true, also passed as true.
        arrow_Ex_plot = np.zeros_like(arrow_Ex)
        arrow_Ey_plot = np.zeros_like(arrow_Ey)
        nonzero = arrow_Emag > 1e-14
        arrow_Ex_plot[nonzero] = arrow_Ex[nonzero] / arrow_Emag[nonzero]
        arrow_Ey_plot[nonzero] = arrow_Ey[nonzero] / arrow_Emag[nonzero]
    else:
        arrow_Ex_plot = arrow_Ex
        arrow_Ey_plot = arrow_Ey

    plt.figure(figsize=(6, 5))
    q = plt.quiver( arrow_coords[:, 0], arrow_coords[:, 1], arrow_Ex_plot, arrow_Ey_plot, 
    arrow_Emag, cmap="coolwarm", angles="xy", scale_units="xy", scale=25, width=0.004)

    cbar = plt.colorbar(q)
    cbar.set_label(r"$|E_R|$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()

    filename = f"{name}_phi_{phi_app:.4f}.png"
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def plot_top_surface_stretch(phi_app, F22_field):
    os.makedirs(top_surface_stretch_png_dir, exist_ok=True)

    coords_ref = mesh.coordinates()
    top_indices = [i for i, c in enumerate(coords_ref) if near(c[1], length)]
    top_indices.sort(key=lambda i: coords_ref[i, 0])

    x1 = coords_ref[top_indices, 0]
    lambda2 = F22_field.compute_vertex_values(mesh)[top_indices]

    plt.figure(figsize=(7, 4))
    plt.plot(x1, lambda2, "b-", linewidth=1.5)
    plt.xlabel(r"$x_1$")
    plt.ylabel(r"Stretch $\lambda_2 = F_{22}$")
    plt.title(f"Top surface vertical stretch, $\\phi$ = {phi_app:.4f}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    filename = f"top_surface_stretch_phi_{phi_app:.4f}.png"
    plt.savefig(os.path.join(top_surface_stretch_png_dir, filename), dpi=300)
    plt.close()

def writeResults(phi_app):
    # Displacement vector
    u_Vis = project(u, W2)
    u_Vis.rename("disp", " ")

    # Save displacement components separately
    ux_Vis = project(u_Vis.sub(0), W)
    ux_Vis.rename("u_x", " ")
    uy_Vis = project(u_Vis.sub(1), W)
    uy_Vis.rename("u_y", " ")

    save_colored_vector_arrows(ux_Vis, uy_Vis, W, displacement_png_dir, "disp_colored_arrows",
                    phi_app, max_arrows_per_direction=20, normalize_arrows=True)

    # Pressure
    p_Vis = project(p, W)
    p_Vis.rename("p", " ")

    # Electric potential
    phi_Vis = project(phi, W)
    phi_Vis.rename("phi", " ")

    # Jacobian
    J_Vis = project(J, W)
    J_Vis.rename("J", " ")

    # Stretch components
    F22_Vis = project(F[1, 1], W)

    plot_top_surface_stretch(phi_app, F22_Vis)

    # TOTAL CAUCHY STRESS
    T = Tmat*F.T/J

    T11_Vis = project_and_save_scalar(T[0,0], W, "T11_total", phi_app, total_stress_png_dir)
    T22_Vis = project_and_save_scalar(T[1,1], W, "T22_total", phi_app, total_stress_png_dir)
    T12_Vis = project_and_save_scalar(T[0,1], W, "T12_total", phi_app, total_stress_png_dir)

    # MAXWELL CAUCHY STRESS
    T11_Maxw_Vis = project_and_save_scalar(T_maxw[0,0], W, "T11_maxw", phi_app, maxw_stress_png_dir)
    T22_Maxw_Vis = project_and_save_scalar(T_maxw[1,1], W, "T22_maxw", phi_app, maxw_stress_png_dir)
    T12_Maxw_Vis = project_and_save_scalar(T_maxw[0,1], W, "T12_maxw", phi_app, maxw_stress_png_dir)

    # CAUCHY STRESS COMPONENTS
    T11_mech_Vis = project_and_save_scalar(T_mech[0,0], W, "T11_mech", phi_app, cauchy_stress_png_dir)
    T22_mech_Vis = project_and_save_scalar(T_mech[1,1], W, "T22_mech", phi_app, cauchy_stress_png_dir)
    T12_mech_Vis = project_and_save_scalar(T_mech[0,1], W, "T12_mech", phi_app, cauchy_stress_png_dir)

    # ELECTRIC FIELD IN REFERENCE CONFIGURATION
    E_R = -pe_grad_scalar(phi)

    save_colored_vector_arrows(E_R[0], E_R[1], W, Eref_png_dir, "E_R_colored_arrows",
                    phi_app, max_arrows_per_direction=20, normalize_arrows=True)

    E_Rx_Vis = project(E_R[0], W)
    E_Ry_Vis = project(E_R[1], W)

    return {
        "u": u_Vis,
        "p": p_Vis,
        "phi": phi_Vis,
        "J": J_Vis,
        "F22": F22_Vis,
        "T11_total": T11_Vis,
        "T22_total": T22_Vis,
        "T12_total": T12_Vis,
        "T11_maxw": T11_Maxw_Vis,
        "T22_maxw": T22_Maxw_Vis,
        "T12_maxw": T12_Maxw_Vis,
        "T11_mech": T11_mech_Vis,
        "T22_mech": T22_mech_Vis,
        "T12_mech": T12_mech_Vis,
        "E_R_x": E_Rx_Vis,
        "E_R_y": E_Ry_Vis,
    }

def save_center_node_field_values_separate_files(phi_app):
    os.makedirs(center_node_values_dir, exist_ok=True)

    hx = length/N
    hy = length/N

    # Total Cauchy stress
    T = Tmat*F.T/J

    # Reference electric field
    E_R = -pe_grad_scalar(phi)

    # Project all scalar fields that we want to export
    fields = {
        "u_x": project(u[0], W),
        "u_y": project(u[1], W),

        "p": project(p, W),
        "phi": project(phi, W),
        "J": project(J, W),

        "T11_total": project(T[0,0], W),
        "T22_total": project(T[1,1], W),
        "T12_total": project(T[0,1], W),

        "T11_mech": project(T_mech[0,0], W),
        "T22_mech": project(T_mech[1,1], W),
        "T12_mech": project(T_mech[0,1], W),

        "T11_maxw": project(T_maxw[0,0], W),
        "T22_maxw": project(T_maxw[1,1], W),
        "T12_maxw": project(T_maxw[0,1], W),

        "E_R1": project(E_R[0], W),
        "E_R2": project(E_R[1], W),
    }

    # Center-node coordinates of the crossed mesh rectangles
    center_coords = []
    for j in range(N):
        for i in range(N):
            center_coords.append(((i + 0.5) * hx, (j + 0.5) * hy))

    # Save one file for each field
    for field_name, field_fun in fields.items():
        rows = []
        for x_c, y_c in center_coords:
            value = field_fun(Point(x_c, y_c))
            rows.append([x_c, y_c, value])
        rows = np.array(rows)

        filename = f"{field_name}_center_nodes_phi_{phi_app:.4f}.txt"
        output_path = os.path.join(center_node_values_dir, filename)

        np.savetxt(
            output_path,
            rows,
            header="x y value",
            comments="",
            fmt="%.10e %.10e %.10e",
        )

    print("Saved separate center-node field files to:")
    print(os.path.abspath(center_node_values_dir))


from datetime import datetime

print("------------------------------------")
print("Start Simulation")
print("------------------------------------")
# Store start time
startTime = datetime.now()

# Boundary conditions
bcs_0 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)  # u1 fix - Left
bcs_1 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2)  # u2 fix - Bottom
#
bcs_2 = DirichletBC(ME.sub(2), 0, facets, 2)  # phi ground - Bottom
bcs_3 = DirichletBC(ME.sub(2), phiTop, facets, 4)  # fixed phi - Top

# BC set
bcs = [bcs_0, bcs_1, bcs_2, bcs_3]

# Set up the non-linear problem
electrostaticProblem = NonlinearVariationalProblem(Res, w, bcs, J=a)

# Set up the non-linear solver
solver  = NonlinearVariationalSolver(electrostaticProblem)

#Solver parameters
prm = solver.parameters
prm['nonlinear_solver'] = 'newton'
prm['newton_solver']['linear_solver'] = "mumps"
prm['newton_solver']['absolute_tolerance']   = 1.e-8
prm['newton_solver']['relative_tolerance']   = 1.e-7
prm['newton_solver']['maximum_iterations']   = 125
prm['newton_solver']['relaxation_parameter'] = 0.5
prm['newton_solver']['error_on_nonconvergence'] = True

try:
    (iterations, converged) = solver.solve()
except RuntimeError as e:
    print("Solver failed:", e)
    converged = False
    iterations = 0

if not converged:
    print("-------------------------------------------")
    print("Newton did not converge.")
    print(f"Iterations used: {iterations}")
    print("Stopping before writing output.")
    print("-------------------------------------------")
else:
    w_array = w.vector().get_local()
    if not np.all(np.isfinite(w_array)):
        print("-------------------------------------------")
        print("NaN or Inf detected in solution.")
        print("Stopping before writing output.")
        print("-------------------------------------------")
    else:
        field_dict = writeResults(phi_app)
        save_center_node_field_values_separate_files(phi_app)

        uy_top = w.sub(0).sub(1)(length/2, length)
        stretch = uy_top/length + 1.0
        current_area = assemble(J*dx)

        print("-------------------------------------------")
        print(f"Applied potential (amplitude): {phi_app:.6f}")
        print(f"Vertical stretch at top midpoint: {stretch:.6f}")
        print(f"Current area: {current_area:.6f}")
        print(f"Newton iterations: {iterations}")
        print("-------------------------------------------")

# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("Elapsed real time: {}".format(elapseTime))