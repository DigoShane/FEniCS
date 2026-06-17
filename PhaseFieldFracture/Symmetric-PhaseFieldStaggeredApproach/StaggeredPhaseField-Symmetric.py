from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pygmsh

import gmsh
import meshio

import os
import shutil

# Reset results folder
results_dir = "results"
if os.path.exists(results_dir):
    shutil.rmtree(results_dir)
os.makedirs(results_dir)

# ------------------------------
#          MESH GENERATION
# ------------------------------
gmsh.initialize()
gmsh.model.add("half_rect_with_hole")

gmsh.model.occ.addRectangle(0, 0, 0, 0.5, 1.0, tag=1)

# Circular hole: center (0.5, 0.5), radius 0.2
gmsh.model.occ.addDisk(0.5, 0.5, 0, 0.2, 0.2, tag=2)

# Cut (subtract hole from rectangle)
gmsh.model.occ.cut([(2, 1)], [(2, 2)])

gmsh.model.occ.synchronize()

# Mesh size control
lc = 0.0025

gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc / 4)

# Deterministic smoother meshing
gmsh.option.setNumber("Mesh.Algorithm", 6)
gmsh.option.setNumber("Mesh.RandomSeed", 1)

# Generate 2D mesh
gmsh.model.mesh.generate(2)

# Optional: save raw .msh file for inspection
gmsh.write("half_rect_with_hole.msh")

# Get mesh data
node_tags, coord, _ = gmsh.model.mesh.getNodes()
coord = coord.reshape(-1, 3)[:, :2]

elem_tags, node_tags_elem = gmsh.model.mesh.getElementsByType(2)
cells = node_tags_elem.reshape(-1, 3) - 1

gmsh.finalize()

# Convert to DOLFIN mesh using MeshEditor
mesh = Mesh()
editor = MeshEditor()
editor.open(mesh, "triangle", 2, 2)
editor.init_vertices(len(coord))
editor.init_cells(len(cells))

for i, pt in enumerate(coord):
    editor.add_vertex(i, pt)

for i, cell in enumerate(cells):
    editor.add_cell(i, cell.astype(np.uintp))

editor.close()

print("Mesh created successfully with", mesh.num_cells(), "cells")

# Visualize Mesh
plt.figure(figsize=(8,8))
plot(mesh)
plt.title("Half-domain mesh")
plt.show()

#-----------------------------------------------------------------------------------

# Function Spaces
Vu = VectorFunctionSpace(mesh, "CG", 1)
Vd = FunctionSpace(mesh, "CG", 1)
Vs = TensorFunctionSpace(mesh, "DG", 0)

u     = Function(Vu, name="Displacement")
d     = Function(Vd, name="Damage")
d_old = Function(Vd, name="Damage_old")

# Material properties
E, nu = 200, 0.2

lmbda = Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))
mu    = Constant(E / (2 * (1 + nu)))

kappa = Constant(lmbda + 2.0/3.0 * mu)

kres = Constant(1e-6)
Gc   = Constant(1.0)
l0   = Constant(0.02)

# Boundary conditions
boundaries = MeshFunction( "size_t", mesh, mesh.topology().dim()-1, 0 )

class TopBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 1.0) and on_boundary

TopBoundary().mark(boundaries, 1)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

def top(x, on_boundary):
    return near(x[1], 1.0) and on_boundary

# Symmetry boundary at x=0.5
def symmetry(x, on_boundary):
    return near(x[0], 0.5) and on_boundary

def internal(x, on_boundary):
    return near( (x[0] - 0.5)**2 + (x[1] - 0.5)**2, 0.2**2, 1e-3 ) and on_boundary

# displacement controlled loading
Uimp = Expression(("0", "t"), t=0.0, degree=0)

# Symmetry BC: u_x = 0 on x=0.5
bc_sym = DirichletBC( Vu.sub(0), Constant(0.0), symmetry)

# Fixed hole
bc_internal = DirichletBC( Vu, Constant((0, 0)), internal )

# Top loading
bc_top = DirichletBC( Vu, Uimp, top )

bcu = [ bc_internal, bc_top, bc_sym ]


#-----------------------------------------------------------------------------------
# Kinematics

def eps(v):
    return sym(grad(v))

def tr_pos(A):
    return (tr(A) + abs(tr(A))) / 2.0

def tr_neg(A):
    return (tr(A) - abs(tr(A))) / 2.0

def psi_plus(v):

    e = eps(v)

    e_dev = e - (1.0/3.0) * tr(e) * Identity(2)

    return (
        (kappa / 2.0) * tr_pos(e)**2
        +
        mu * inner(e_dev, e_dev)
    )

def psi_minus(v):
    return (kappa / 2.0) * tr_neg(eps(v))**2

def degradation(phi):
    return (1.0 - phi)**2 + kres

#History field
Vh = FunctionSpace(mesh, "DG", 0)
H = Function(Vh, name="HistoryFunction")

def update_history():
    W_plus = project(psi_plus(u), Vh)
    H.vector()[:] = np.maximum( H.vector().get_local(), W_plus.vector().get_local() )


#--------------------------------------------------------------------------------------
# Variational Forms

du = TrialFunction(Vu)
vu = TestFunction(Vu)

def sigma_degraded(v, phi):
    e = eps(v)
    e_vol = tr(e)
    e_vol_pos = (e_vol + abs(e_vol)) / 2.0
    e_vol_neg = (e_vol - abs(e_vol)) / 2.0
    e_dev = e - (1.0/3.0) * e_vol * Identity(2)

    sig_vol_pos = kappa * e_vol_pos * Identity(2)
    sig_vol_neg = kappa * e_vol_neg * Identity(2)
    sig_dev = 2.0 * mu * e_dev

    return ( degradation(phi) * (sig_vol_pos + sig_dev) + sig_vol_neg )

F_u = inner( sigma_degraded(u, d), eps(vu) ) * dx
dF_u = derivative(F_u, u, du)


#--------------------------------------------------------------------------------------
# Nonlinear solver

problem_u = NonlinearVariationalProblem( F_u, u, bcs=bcu, J=dF_u )

solver_u = NonlinearVariationalSolver(problem_u)

prm = solver_u.parameters

prm["newton_solver"]["absolute_tolerance"] = 1e-5
prm["newton_solver"]["relative_tolerance"] = 1e-4
prm["newton_solver"]["maximum_iterations"] = 100
prm["newton_solver"]["linear_solver"]      = "mumps"
prm["newton_solver"]["report"]             = True
prm["newton_solver"]["relaxation_parameter"] = 0.5


#--------------------------------------------------------------------------------------
# Damage solve

dd = TrialFunction(Vd)
q  = TestFunction(Vd)

def build_damage_forms():

    a = ( (Gc/l0 + 2.0*H) * dd * q  +  Gc * l0 * dot(grad(dd), grad(q)) ) * dx
    L = 2.0 * H * q * dx

    return a, L

def solve_displacement():
    solver_u.solve()

def solve_damage():

    a_d, L_d = build_damage_forms()

    solve( a_d == L_d, d, solver_parameters={"linear_solver": "lu"} )

    d.vector()[:] = np.maximum( d.vector().get_local(), d_old.vector().get_local() )

    d.vector()[:] = np.clip( d.vector().get_local(), 0.0, 1.0 )


#--------------------------------------------------------------------------------------
# Energy functionals

def stored_energy():

    return assemble( ( degradation(d) * psi_plus(u) + psi_minus(u) ) * dx )

def dissipated_energy():

    return assemble( ( Gc/(2*l0) * d**2 + Gc*l0/2 * dot(grad(d), grad(d)) ) * dx )


#--------------------------------------------------------------------------------------
# ParaView output

xdmf_u = XDMFFile("phase_field_half_domain_displacement.xdmf")
xdmf_d = XDMFFile("phase_field_half_domain_damage.xdmf")

for f in [xdmf_u, xdmf_d]:
    f.parameters["flush_output"] = True
    f.parameters["functions_share_mesh"] = True


#--------------------------------------------------------------------------------------
# Load stepping

tol, Nitermax = 1e-3, 500

loading = np.concatenate((np.linspace(0, 70e-3, 6), np.linspace(70e-3, 225e-3, 80)[1:]))

N_steps = loading.shape[0]

results = np.zeros((N_steps, 3))

#Stress tracking arrays
sigma_tip_xx  = np.zeros(N_steps)
sigma_tip_yy  = np.zeros(N_steps)
sigma_tip_xy  = np.zeros(N_steps)

sigma_left_xx = np.zeros(N_steps)
sigma_left_yy = np.zeros(N_steps)
sigma_left_xy = np.zeros(N_steps)


#points of interest for stress:
tip_point = Point(0.5, 0.7) #top of circle
left_point = Point(0.3, 0.5) #left end of circle

for i, t in enumerate(loading):

    print( "Time step: {}  (u_imp = {:.4f})".format(i+1, t) )

    Uimp.t = t

    res = 1.0
    j   = 1

    while res > tol and j < Nitermax:
        solve_displacement()
        update_history()
        d_old.assign(d)
        solve_damage()
        res = np.max( d.vector().get_local() - d_old.vector().get_local() )

        print("   Iteration {:3d}:  max(Δd) = {:.2e}".format(j, res))
        j += 1

    #------------------------------------------
    # Post-processing

    n = FacetNormal(mesh)
    traction = dot(sigma_degraded(u, d), n)

    reaction = assemble( dot(traction, as_vector((0,1))) * ds(1) )

    results[i,0] = reaction
    results[i,1] = stored_energy()
    results[i,2] = dissipated_energy()
    
    sigma_proj = project(sigma_degraded(u, d),Vs)

    # Evaluate stresses at tip
    sigma_tip = sigma_proj(tip_point)
    
    sigma_tip_xx[i] = sigma_tip[0]
    sigma_tip_xy[i] = sigma_tip[1]
    sigma_tip_yy[i] = sigma_tip[3]
    
    # Evaluate stresses at left point
    sigma_left = sigma_proj(left_point)
    
    sigma_left_xx[i] = sigma_left[0]
    sigma_left_xy[i] = sigma_left[1]
    sigma_left_yy[i] = sigma_left[3]

    xdmf_u.write(u, t)
    xdmf_d.write(d, t)

    clear_output(wait=True)

    plt.figure(figsize=(6,6))

    p = plot(d, vmin=0, vmax=1)

    plt.colorbar(p)

    plt.title("Damage  t={:.4f}".format(t))

    plt.savefig("./results/phase_field_{:04d}.png".format(i), dpi=400)

    plt.close()

xdmf_u.close()
xdmf_d.close()


#--------------------------------------------------------------------------------
# Summary plots

# force vs displacement
plt.figure()
plt.plot(loading, results[:,0], "-o")
plt.xlabel("Imposed displacement")
plt.ylabel("Vertical force")
plt.title("Load-displacement curve")
plt.show()

# Energy evolution versus displacement
plt.figure()
plt.plot(loading, results[:,1], label="elastic energy")
plt.plot(loading, results[:,2],label="fracture energy")
plt.plot(loading, results[:,1] + results[:,2], label="total energy")
plt.xlabel("Imposed displacement")
plt.ylabel("Energies")
plt.legend()
plt.title("Energy evolution")
plt.show()

#Stress evolution at crack tip
plt.figure(figsize=(8,6))
plt.plot(loading, sigma_tip_xx, label=r'$\sigma_{xx}$')
plt.plot(loading, sigma_tip_yy, label=r'$\sigma_{yy}$')
plt.plot(loading, sigma_tip_xy, label=r'$\sigma_{xy}$')
plt.xlabel("Applied displacement")
plt.ylabel("Stress at crack tip")
plt.title("Stress evolution at crack tip")
plt.legend()
plt.grid(True)
plt.savefig("stress_tip_vs_displacement.png", dpi=400)
plt.show()


# Stress evolution at left end of circle
plt.figure(figsize=(8,6))
plt.plot(loading, sigma_left_xx, label=r'$\sigma_{xx}$')
plt.plot(loading, sigma_left_yy, label=r'$\sigma_{yy}$')
plt.plot(loading, sigma_left_xy, label=r'$\sigma_{xy}$')
plt.xlabel("Applied displacement")
plt.ylabel("Stress at left end")
plt.title("Stress evolution at left end of circle")
plt.legend()
plt.grid(True)
plt.savefig("stress_left_vs_displacement.png", dpi=400)
plt.show()


#--------------------------------------------------------------------------------
# Save results


#top of circle
tip_data = np.column_stack(( loading, sigma_tip_xx, sigma_tip_yy, sigma_tip_xy))
np.savetxt("stress_tip_vs_displacement.txt", tip_data, header="displacement sigma_xx sigma_yy sigma_xy")

#left of circle
left_data = np.column_stack(( loading, sigma_left_xx, sigma_left_yy, sigma_left_xy))

np.savetxt("stress_left_vs_displacement.txt", left_data, header="displacement sigma_xx sigma_yy sigma_xy")

