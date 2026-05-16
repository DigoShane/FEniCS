from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import pygmsh

import gmsh
#import meshio

import os
import shutil

# -------------------------------------------------
# Create output directories
# -------------------------------------------------

base_dir = "results"

subdirs = [ "damage", "sxx", "syy", "sxy", "von", "sigma1", "sigma2", "damage_contours" ]

# Remove previous results
if os.path.exists(base_dir):
    shutil.rmtree(base_dir)

os.makedirs(base_dir)

# Create subdirectories
for s in subdirs:
    os.makedirs(os.path.join(base_dir, s))


L, Ht = 1.0, 1.0
a = 0.2   # crack length
eps_geom = 1e-3


gmsh.initialize()
gmsh.model.add("SENT")

# Rectangle
rect = gmsh.model.occ.addRectangle(0, 0, 0, L, Ht, tag=1)

# Crack: small slit
crack = gmsh.model.occ.addRectangle(0, Ht/2 - eps_geom, 0, a, 2*eps_geom, tag=2)

gmsh.model.occ.cut([(2, rect)], [(2, crack)], removeObject=True, removeTool=True)
gmsh.model.occ.synchronize()


#mesh size control
lc = 0.004
gmsh.option.setNumber("Mesh.CharacteristicLengthMax", lc)
gmsh.option.setNumber("Mesh.CharacteristicLengthMin", lc / 4)

gmsh.model.mesh.generate(2) 

gmsh.write("SENT.msh")

node_tags, coord, _ = gmsh.model.mesh.getNodes() #_ is parametric coordinates
coord = coord.reshape(-1, 3)[:, :2]

elem_tags, node_tags_elem = gmsh.model.mesh.getElementsByType(2)
cells = node_tags_elem.reshape(-1, 3) - 1

gmsh.finalize()

# ------------------------------
# Convert to FEniCS mesh
# ------------------------------
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
#-----------------------------------------------------------------------------------


#Function Spaces
Vu = VectorFunctionSpace(mesh, "CG", 1)   # displacement
Vd = FunctionSpace(mesh,       "CG", 1)   # damage / phase-field
Vs = TensorFunctionSpace(mesh, "DG", 0)   # stress (for output only)
 
u    = Function(Vu, name="Displacement")
d    = Function(Vd, name="Damage")
d_old = Function(Vd, name="Damage_old")   # previous iterate (stagger convergence)

# Material properties
E, nu = 200, 0.2
lmbda  = Constant(E * nu / ((1 + nu) * (1 - 2 * nu)))
mu     = Constant(E / (2 * (1 + nu)))
kappa  = Constant(lmbda + 2.0/3.0 * mu)   # bulk modulus

kres  = Constant(1e-6)          # residual stiffness
Gc    = Constant(2.7)           # critical energy release rate
l0    = Constant(0.02)          # phase-field length scale
tol_geom = 5e-4   # adjust based on mesh

tip = np.array([a, Ht/2])

# Boundary conditions
boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)

def bottom(x, on_boundary):
    return near(x[1], 0.0) and on_boundary

def top(x, on_boundary):
    return near(x[1], Ht) and on_boundary

top_id = 1
bottom_id = 2

AutoSubDomain(top).mark(boundaries, top_id)
AutoSubDomain(bottom).mark(boundaries, bottom_id)

ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

def bottom(x, on_boundary):

#disp controlled loading.
Uimp = Expression(("0", "t"), t=0.0, degree=0)

#disp fixed on bottom bdry. Top bdry has imposed vertical displacement Uimp.
bcu = [ DirichletBC(Vu, Constant((0, 0)), bottom),
        DirichletBC(Vu, Uimp, top)]

def crack(x, on_boundary):
    return (
        (
            abs(x[1] - (Ht/2 - eps_geom)) < tol_geom and x[0] <= a + tol_geom
        ) or (
            abs(x[1] - (Ht/2 + eps_geom)) < tol_geom and x[0] <= a + tol_geom
        ) or (
            abs(x[0] - a) < tol_geom and abs(x[1] - Ht/2) <= eps_geom + tol_geom
        )
    ) and on_boundary

bcd = DirichletBC(Vd, Constant(1.0), crack)

#Visualizing the d=1 BC on the slit.
bc_values = Function(Vd)
bcd.apply(bc_values.vector())

c = plot(bc_values)
plt.title("BC applied: should be 1 on crack")
plt.colorbar(c)
plt.show()


#Kinematics — Volumetric/Deviatoric Split
def eps(v):
    return sym(grad(v))
 
def tr_pos(A):
    return (tr(A) + abs(tr(A))) / 2.0
 
def tr_neg(A):
    return (tr(A) - abs(tr(A))) / 2.0
 
def psi_plus(v):
    e    = eps(v)
    e_dev = e - (1.0/3.0) * tr(e) * Identity(2)   # deviatoric strain (2D)
    return (kappa / 2.0) * tr_pos(e)**2 + mu * inner(e_dev, e_dev)
 
def psi_minus(v):
    return (kappa / 2.0) * tr_neg(eps(v))**2
 
def degradation(phi):
    return (1.0 - phi)**2 + kres
 
# History Field H
Vh = FunctionSpace(mesh, "DG", 0)
H  = Function(Vh, name="HistoryFunction")
 
def update_history():
    W_plus = project(psi_plus(u), Vh)
    H.vector()[:] = np.maximum(H.vector().get_local(),
                               W_plus.vector().get_local())

# Reaction Force
def reaction_force_y():
    n = FacetNormal(mesh)
    traction = dot(sigma_degraded(u, d), n)
    Fy = assemble(traction[1] * ds(top_id))
    return Fy

#----------------------------------------------------------------------------------------
#Variational Forms
 
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
    sig_dev     = 2.0 * mu * e_dev

    return degradation(phi) * (sig_vol_pos + sig_dev) + sig_vol_neg

# Bilinear and linear forms for u (linear in du for fixed d)
F_u = inner(sigma_degraded(u, d), eps(vu)) * dx

dF_u = derivative(F_u, u, du)

# Nonlinear solver for displacement
problem_u = NonlinearVariationalProblem(F_u, u, bcs=bcu, J=dF_u)
solver_u  = NonlinearVariationalSolver(problem_u)

# Typical parameters (adjust as needed)
prm = solver_u.parameters
prm["newton_solver"]["absolute_tolerance"] = 1e-8
prm["newton_solver"]["relative_tolerance"] = 1e-7
prm["newton_solver"]["maximum_iterations"] = 25
prm["newton_solver"]["linear_solver"]      = "mumps"   # or "lu", "superlu_dist"
prm["newton_solver"]["report"]             = True

dd  = TrialFunction(Vd) 
q   = TestFunction(Vd)
 
def build_damage_forms():
    a = ( (Gc/l0 + 2.0*H) * dd * q
        + Gc * l0 * dot(grad(dd), grad(q)) ) * dx
    L = 2.0 * H * q * dx
    return a, L
 
 
def solve_displacement():
    solver_u.solve()

def solve_damage():
    a_d, L_d = build_damage_forms()
    solve(a_d == L_d, d, bcs=[bcd],solver_parameters={"linear_solver": "lu"})          # or "mumps"
    d.vector()[:] = np.maximum(d.vector().get_local(), d_old.vector().get_local()) # pointwise maximum to enforce irreversibilitya
    d.vector()[:] = np.clip(d.vector().get_local(), 0.0, 1.0) #restricts to [0,1].
 
#Energy functionals
def stored_energy():
    return assemble( (degradation(d) * psi_plus(u) + psi_minus(u)) * dx )
 
def dissipated_energy():
    return assemble( (Gc/(2*l0) * d**2 + Gc*l0/2 * dot(grad(d), grad(d))) * dx )
 
#ParaView Output
xdmf_u = XDMFFile("phase_field_no_mfront_displacement.xdmf")
xdmf_d = XDMFFile("phase_field_no_mfront_damage.xdmf")
 
for f in [xdmf_u, xdmf_d]:
    f.parameters["flush_output"]         = True
    f.parameters["functions_share_mesh"] = True


#Load-Stepping Loop
tol, Nitermax = 1e-3, 500

loading = np.concatenate((np.linspace(0,   70e-3,  12), np.linspace(70e-3, 500e-3, 56)[1:]))   # skip first zero if you want
N_steps = loading.shape[0]
results = np.zeros((N_steps, 4))# 0 -> force, 1 -> elastic energy, 2 -> fracture energy, 3 -> max damage
 
#tip stress components
tip_sxx = np.zeros(N_steps)
tip_syy = np.zeros(N_steps)
tip_sxy = np.zeros(N_steps)
tip_vm  = np.zeros(N_steps)
 
for i, t in enumerate(loading):
    print("Time step: {}  (u_imp = {:.4f})".format(i+1, t))
    Uimp.t = t
 
    # ---- Alternate minimization ----
    res = 1.0
    j   = 1
    while res > tol and j < Nitermax:
        # Step A: solve mechanics with current d
        solve_displacement()
 
        # Update history field H = max(H, W+(u))
        update_history()
 
        # Step B: solve damage with current u and H
        d_old.assign(d)
        solve_damage()
 
        # Convergence: max pointwise damage increment
        res = np.max(d.vector().get_local() - d_old.vector().get_local())
        print("   Iteration {:3d}:  max(Δd) = {:.2e}".format(j, res))
        j += 1

    stress = sigma_degraded(u,d)
    sxx = project(stress[0,0], Vd)
    syy = project(stress[1,1], Vd)
    sxy = project(stress[0,1], Vd)

    von_expr = sqrt(abs(sxx*sxx - sxx*syy + syy*syy + 3.0*sxy*sxy))
    von = project(von_expr, Vd)

    avg_stress = 0.5 * (sxx + syy)
    radius = sqrt( abs( ((sxx - syy)/2.0)**2 + sxy*sxy ))

    sigma1_expr = avg_stress + radius
    sigma2_expr = avg_stress - radius

    sigma1 = project(sigma1_expr, Vd)
    sigma2 = project(sigma2_expr, Vd)

    # ---- Post-processing ----
    results[i,0] = reaction_force_y()
    results[i,1] = stored_energy()
    results[i,2] = dissipated_energy()
    results[i,3] = d.vector().max()
 
    xdmf_u.write(u, t)
    xdmf_d.write(d, t)
 
    clear_output(wait=True)
    plt.figure()
    p = plot(d, vmin=0, vmax=1)
    plt.colorbar(p)
    plt.title("Damage  t={:.4f}".format(t))
    plt.savefig("./results_d/phase_field_{:04d}.png".format(i), dpi=400)
    plt.close()

    plt.figure()
    p = plot(sxx)
    plt.colorbar(p)
    plt.title(f"sigma_xx t={t:.4f}")
    plt.savefig(f"./results_sxx/sxx_{i:04d}.png")
    plt.close()

    plt.figure()
    p = plot(syy)
    plt.colorbar(p)
    plt.title(f"sigma_yy t={t:.4f}")
    plt.savefig(f"./results_syy/syy_{i:04d}.png")
    plt.close()

    plt.figure()
    p = plot(sxy)
    plt.colorbar(p)
    plt.title(f"sigma_xy t={t:.4f}")
    plt.savefig(f"./results_sxy/sxy_{i:04d}.png")
    plt.close()

    plt.figure()
    p = plot(von)
    plt.colorbar(p)
    plt.title(f"von Mises t={t:.4f}")
    plt.savefig(f"./results_von/von_{i:04d}.png")
    plt.close()

    plt.figure()
    p = plot(sigma1)
    plt.colorbar(p)
    plt.title(f"sigma_1 t={t:.4f}")
    plt.savefig(f"./results/sigma1/sigma1_{i:04d}.png")
    plt.close()

    plt.figure()
    p = plot(sigma2)
    plt.colorbar(p)
    plt.title(f"sigma_2 t={t:.4f}")
    plt.savefig(f"./results/sigma2/sigma2_{i:04d}.png")
    plt.close()

    #computing the stress at the tip:
    tip_x = a + 2*lc
    tip_y = Ht/2
    tip_sxx[i] = sxx(tip_x, tip_y)
    tip_syy[i] = syy(tip_x, tip_y)
    tip_sxy[i] = sxy(tip_x, tip_y)
    tip_vm[i]  = von(tip_x, tip_y)

xdmf_u.close()
xdmf_d.close()
 
#Summary Plots
plt.figure()
plt.plot(loading, results[:, 0], "-o")
plt.xlabel("Imposed displacement")
plt.ylabel("Vertical force")
plt.title("Load-displacement curve")
plt.show()
 
plt.figure()
plt.plot(loading, results[:, 1], label="elastic energy")
plt.plot(loading, results[:, 2], label="fracture energy")
plt.plot(loading, results[:, 1] + results[:, 2], label="total energy")
plt.xlabel("Imposed displacement")
plt.ylabel("Energies")
plt.legend()
plt.title("Energy evolution")
plt.show()

plt.figure()
plt.plot(loading, tip_sxx, label="sigma_xx")
plt.plot(loading, tip_syy, label="sigma_yy")
plt.plot(loading, tip_sxy, label="sigma_xy")
plt.plot(loading, tip_vm, label="von mises")
plt.xlabel("Displacement")
plt.ylabel("Stress at crack tip")
plt.legend()
plt.show()

max_force = np.max(np.abs(results[:,0]))
print("Maximum force =", max_force)

print("Max |sigma_xx| at tip =", np.max(np.abs(tip_sxx)))
print("Max |sigma_yy| at tip =", np.max(np.abs(tip_syy)))
print("Max |sigma_xy| at tip =", np.max(np.abs(tip_sxy)))
print("Max |von Mises| at tip =", np.max(np.abs(tip_vm)))

force = results[:,0]

tol_force = 1e-3 * np.max(np.abs(force))

peak_idx = np.argmax(np.abs(force))

indices = np.where(
    np.abs(force[peak_idx:]) < tol_force
)[0]

if len(indices) > 0:
    idx = peak_idx + indices[0]
    print("Force drops to ~0 at displacement =", loading[idx])