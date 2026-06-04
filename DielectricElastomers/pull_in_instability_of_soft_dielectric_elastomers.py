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
vareps   = vareps_r*vareps_0         #  permittivity of the material

# Simulation time control-related params
t        = 0.0           # start time (s)
rampRate = float(input("Enter the ramp rate: ")) # s^{-1}
Ttot     = 1.0/rampRate  # total simulation time (s)
numSteps = int(input("Enter the number of steps: "))
dt       = Ttot/numSteps       # (fixed) step size
dk       = Constant(dt)

# Normalization parameter for voltage is l*sqrt(Geq_0/vareps)
phiTot = 1.25* float(length*np.sqrt(float(Geq_0)/float(vareps)))  # final normalized value of phi

# Electric potential scale:
phi_scale = float(length*np.sqrt(float(Geq_0)/float(vareps)))
# Electric field scale:
E_scale = float(np.sqrt(float(Geq_0)/float(vareps)))
# Stress scale:
stress_scale = float(Geq_0)
# The final nominal normalized voltage is 1.25 in the present code
phi_factor = phiTot/phi_scale

# Boundary condition to ramp up electrostatic potential
phiRamp = Expression(("phi_tot*t/Ttot"),
                      t = 0.0, phi_tot = phiTot, Ttot=Ttot, degree=1)

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

all_png_dirs = [ total_stress_png_dir, maxw_stress_png_dir, cauchy_stress_png_dir, Eref_png_dir]

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


def save_scalar_png(scalar_function, folder, name, t, vmin_plot=None, vmax_plot=None):
    os.makedirs(folder, exist_ok=True)

    coords = mesh.coordinates()
    cells = mesh.cells()
    values = scalar_function.compute_vertex_values(mesh)

    if not np.all(np.isfinite(values)):
        print(f"Skipping {name} at t = {t:.4f}: NaN or Inf detected.")
        return

    vmin = np.min(values)
    vmax = np.max(values)

    plt.figure(figsize=(6, 4))
    p_plot = plt.tripcolor( coords[:, 0], coords[:, 1], values, triangles=cells, shading="gouraud", vmin=vmin_plot, vmax=vmax_plot)
    plt.colorbar(p_plot)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.tight_layout()

    filename = f"{name}_t_{t:.4f}.png"
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def project_and_save_scalar(expr, W, name, t, folder, vmin_plot=None, vmax_plot=None):
    field = project(expr, W)
    field.rename(name, " ")
    save_scalar_png(field, folder, name, t, vmin_plot, vmax_plot)

    return field

def save_colored_vector_arrows(expr_x, expr_y, W, folder, name, t, vector_scale=1.0, max_arrows_per_direction=20, normalize_arrows=True, vmin_plot=None, vmax_plot=None):
    os.makedirs(folder, exist_ok=True)

    Ex_fun = project(expr_x/vector_scale, W)
    Ey_fun = project(expr_y/vector_scale, W)
    coords = mesh.coordinates()
    Ex = Ex_fun.compute_vertex_values(mesh)
    Ey = Ey_fun.compute_vertex_values(mesh)

    if not np.all(np.isfinite(Ex)) or not np.all(np.isfinite(Ey)):
        print(f"Skipping {name} at t = {t:.4f}: NaN or Inf detected.")
        return

    Emag = np.sqrt(Ex**2 + Ey**2)

    if not np.all(np.isfinite(Emag)):
        print(f"Skipping {name} at t = {t:.4f}: NaN or Inf detected in magnitude.")
        return

    num_vertices = coords.shape[0]
    target_arrows = max_arrows_per_direction**2
    stride = max(1, int(num_vertices / target_arrows))

    arrow_coords = coords[::stride]
    arrow_Ex = Ex[::stride]
    arrow_Ey = Ey[::stride]
    arrow_Emag = Emag[::stride]

    if normalize_arrows:
        arrow_Ex_plot = np.zeros_like(arrow_Ex)
        arrow_Ey_plot = np.zeros_like(arrow_Ey)
        nonzero = arrow_Emag > 1e-14
        arrow_Ex_plot[nonzero] = arrow_Ex[nonzero] / arrow_Emag[nonzero]
        arrow_Ey_plot[nonzero] = arrow_Ey[nonzero] / arrow_Emag[nonzero]
    else:
        arrow_Ex_plot = arrow_Ex
        arrow_Ey_plot = arrow_Ey

    Emin = np.min(Emag)
    Emax = np.max(Emag)

    plt.figure(figsize=(6, 5))
    q = plt.quiver( arrow_coords[:, 0], arrow_coords[:, 1], arrow_Ex_plot, arrow_Ey_plot, arrow_Emag, cmap="coolwarm", angles="xy", scale_units="xy", scale=25, width=0.004, clim=(vmin_plot, vmax_plot) if vmin_plot is not None and vmax_plot is not None else None)

    if vmin_plot is not None and vmax_plot is not None:
        q.set_clim(vmin_plot, vmax_plot)

    cbar = plt.colorbar(q)
    cbar.set_label(r"$|E_R|/E_0$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.xlim(0.0, length)
    plt.ylim(0.0, length)
    plt.tight_layout()

    filename = f"{name}_t_{t:.4f}.png"
    filepath = os.path.join(folder, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()

def writeResults(t):
    # Displacement vector
    u_Vis = project(u, W2)
    u_Vis.rename("disp", " ")

    # Pressure
    p_Vis = project(p, W)
    p_Vis.rename("p", " ")

    # Electric potential
    phi_Vis = project(phi, W)
    phi_Vis.rename("phi", " ")

    # Jacobian
    J_Vis = project(J, W)
    J_Vis.rename("J", " ")

    # TOTAL CAUCHY STRESS
    T = Tmat*F.T/J

    stress_limit = 3.0

    T11_Vis = project_and_save_scalar(T[0,0]/stress_scale, W, "T11_total_over_G0", t, total_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)
    T22_Vis = project_and_save_scalar(T[1,1]/stress_scale, W, "T22_total_over_G0", t, total_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)
    T12_Vis = project_and_save_scalar(T[0,1]/stress_scale, W, "T12_total_over_G0", t, total_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)

    # MAXWELL CAUCHY STRESS
    T11_Maxw_Vis = project_and_save_scalar(T_maxw[0,0]/stress_scale, W, "T11_maxw_over_G0", t, maxw_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)
    T22_Maxw_Vis = project_and_save_scalar(T_maxw[1,1]/stress_scale, W, "T22_maxw_over_G0", t, maxw_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)
    T12_Maxw_Vis = project_and_save_scalar(T_maxw[0,1]/stress_scale, W, "T12_maxw_over_G0", t, maxw_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)

    # CAUCHY STRESS COMPONENTS
    T11_mech_Vis = project_and_save_scalar( T_mech[0,0]/stress_scale, W, "T11_cauchy_over_G0", t, cauchy_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)
    T22_mech_Vis = project_and_save_scalar( T_mech[1,1]/stress_scale, W, "T22_cauchy_over_G0", t, cauchy_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)
    T12_mech_Vis = project_and_save_scalar( T_mech[0,1]/stress_scale, W, "T12_cauchy_over_G0", t, cauchy_stress_png_dir, vmin_plot=-stress_limit, vmax_plot= stress_limit)

    # ELECTRIC FIELD IN REFERENCE CONFIGURATION
    E_R = -pe_grad_scalar(phi)
    E_limit = 1.5

    save_colored_vector_arrows( E_R[0], E_R[1], W, Eref_png_dir, "E_R_colored_arrows_over_E0", t, vector_scale=E_scale, max_arrows_per_direction=20, normalize_arrows=True, vmin_plot=0.0, vmax_plot=E_limit)

from datetime import datetime

print("------------------------------------")
print("Start Simulation")
print("------------------------------------")
# Store start time
startTime = datetime.now()

# Give the step a descriptive name
step = "Actuate"

# Boundary conditions
bcs_0 = DirichletBC(ME.sub(0).sub(0), 0, facets, 1)  # u1 fix - Left
bcs_1 = DirichletBC(ME.sub(0).sub(1), 0, facets, 2)  # u2 fix - Bottom
#
bcs_2 = DirichletBC(ME.sub(2), 0, facets, 2)  # phi ground - Bottom
bcs_3 = DirichletBC(ME.sub(2), phiRamp, facets, 4)  # phi ramp - Top

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

# Initalize output array for tip displacement
timeHist0 = []
timeHist1 = []
timeHistTime = []
#Iinitialize a counter for reporting data
ii=0

# Time-stepping solution procedure loop
while t <= Ttot :
    t += float(dk)
    ii += 1
    phiRamp.t = t

    try:
        (iter, converged) = solver.solve()
    except RuntimeError as e:
        print("Solver failed:", e)
        break

    if not converged:
        print("-------------------------------------------")
        print(f"Newton did not converge at increment {ii}, t = {t:.6f}")
        print(f"Iterations used: {iter}")
        print("Stopping before writing corrupted output.")
        print("-------------------------------------------")
        break

    w_array = w.vector().get_local()
    if not np.all(np.isfinite(w_array)):
        print("-------------------------------------------")
        print(f"NaN or Inf detected in solution at increment {ii}, t = {t:.6f}")
        print("Stopping before writing output.")
        print("-------------------------------------------")
        break

    writeResults(t)

    w_old.vector()[:] = w.vector()

    timeHistTime.append(t)
    timeHist0.append(w.sub(0).sub(1)(length, length))
    timeHist1.append(w.sub(2)(length, length))

    # print progress of calculation
    if ii%10 == 0:
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("Step: {} |   Increment: {} | Iterations: {}".format(step, ii, iter))
        print("Simulation Time: {} s | dt: {} s".format(round(t,2), round(dt, 3)))
        print()


# Report elapsed real time for whole analysis
endTime = datetime.now()
elapseTime = endTime - startTime
print("-------------------------------------------")
print("Elapsed real time:  {}".format(elapseTime))
print("-------------------------------------------")

# set plot font to size 14
font = {'size'   : 14}
plt.rc('font', **font)

# Get array of default plot colors
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# Plot the normalized dimensionless quantity for $\phi$ used in Wang et al. 2016
# versus stretch in the vertical direction.
timeHist0 = np.array(timeHist0)
timeHist1 = np.array(timeHist1)
timeHistTime = np.array(timeHistTime)

normVolts = timeHist1/(length * np.sqrt(float(Geq_0)/float(vareps)))
stretch = timeHist0/length + 1.0
#
plt.plot(normVolts, stretch, c=colors[0], linewidth=1.0, marker='.')
# plt.scatter(normVolts[ii-1], stretch[ii-1], c='k', marker='x', s=100)
plt.grid(linestyle="--", linewidth=0.5, color='b')
ax = plt.gca()
#
ax.set_ylabel(r'$\lambda$')
ax.set_ylim([0.2,1.1])
ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
#
ax.set_xlabel(r'$(\phi/\ell_0)/\sqrt{G_0/\varepsilon} $')
ax.set_xlim([0,1.3])
#
from matplotlib.ticker import AutoMinorLocator,FormatStrFormatter
ax.xaxis.set_minor_locator(AutoMinorLocator())
ax.yaxis.set_minor_locator(AutoMinorLocator())
plt.show()

fig = plt.gcf()
fig.set_size_inches(6,4)
plt.tight_layout()
plt.show()