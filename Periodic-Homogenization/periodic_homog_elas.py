from __future__ import print_function
from dolfin import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

a = 1.         # unit cell width
b = sqrt(3.)/2. # unit cell height
c = 0.5        # horizontal offset of top boundary
R = 0.2        # inclusion radius
vol = a*b      # unit cell volume
# we define the unit cell vertices coordinates for later use
vertices = np.array([[0, 0.],
                     [a, 0.],
                     [a+c, b],
                     [c, b]])
fname = "hexag_incl"
mesh = Mesh(fname + ".xml")
subdomains = MeshFunction("size_t", mesh, fname + "_physical_region.xml")
facets = MeshFunction("size_t", mesh, fname + "_facet_region.xml")
plt.figure()
plot(subdomains)
plt.savefig("subdomains.png", dpi=300)
plt.close()

# class used to define the periodic boundary map
class PeriodicBoundary(SubDomain): #inheritence from SubDomain class.
    def __init__(self, vertices, tolerance=DOLFIN_EPS):
        """ vertices stores the coordinates of the 4 unit cell corners"""
        SubDomain.__init__(self, tolerance)# calling parent class.
        self.tol = tolerance
        self.vv = vertices
        self.a1 = self.vv[1,:]-self.vv[0,:] # first vector generating periodicity
        self.a2 = self.vv[3,:]-self.vv[0,:] # second vector generating periodicity
        # check if UC vertices form indeed a parallelogram
        assert np.linalg.norm(self.vv[2, :]-self.vv[3, :] - self.a1) <= self.tol
        assert np.linalg.norm(self.vv[2, :]-self.vv[1, :] - self.a2) <= self.tol
        
    def inside(self, x, on_boundary):
        return bool((near(x[0], self.vv[0,0] + x[1]*self.a2[0]/self.vv[3,1], self.tol) or 
                     near(x[1], self.vv[0,1] + x[0]*self.a1[1]/self.vv[1,0], self.tol)) and 
                     (not ((near(x[0], self.vv[1,0], self.tol) and near(x[1], self.vv[1,1], self.tol)) or 
                     (near(x[0], self.vv[3,0], self.tol) and near(x[1], self.vv[3,1], self.tol)))) and on_boundary)

    def map(self, x, y):
        if near(x[0], self.vv[2,0], self.tol) and near(x[1], self.vv[2,1], self.tol): # if on top-right corner
            y[0] = x[0] - (self.a1[0]+self.a2[0])
            y[1] = x[1] - (self.a1[1]+self.a2[1])
        elif near(x[0], self.vv[1,0] + x[1]*self.a2[0]/self.vv[2,1], self.tol): # if on right boundary
            y[0] = x[0] - self.a1[0]
            y[1] = x[1] - self.a1[1]
        else:   # should be on top boundary
            y[0] = x[0] - self.a2[0]
            y[1] = x[1] - self.a2[1]


Em = 50e3
num = 0.2
Er = 210e3
nur = 0.3

material_parameters = [(Em, num), (Er, nur)]
nphases = len(material_parameters)
def eps(v):
    return sym(grad(v))
def sigma(v, i, Eps):
    E, nu = material_parameters[i]
    lmbda = E*nu/(1+nu)/(1-2*nu)
    mu = E/2/(1+nu)
    return lmbda*tr(eps(v) + Eps)*Identity(2) + 2*mu*(eps(v)+Eps)

def save_displacement_quiver(mesh, subdomains, v_fun, Eps_np, filename,
                             scale_factor=1.0, max_arrows=300):
    X = []
    U = []
    for cell in cells(mesh):
        mp = cell.midpoint()
        x = np.array([mp.x(), mp.y()])
        v_val = np.array(v_fun(mp))
        u_val = v_val
        X.append(x)
        U.append(u_val)

    X = np.array(X)
    U = np.array(U)
    n = len(X)
    if n > max_arrows:
        ids = np.linspace(0, n - 1, max_arrows).astype(int)
        X = X[ids]
        U = U[ids]

    # Plot subdomains as background
    plt.figure(figsize=(8, 5))
    plot(subdomains)
    plt.quiver( X[:, 0], X[:, 1], scale_factor*U[:, 0], scale_factor*U[:, 1], angles="xy", 
                scale_units="xy", scale=0.25, width=0.003)
    plt.axis("equal")
    plt.title("Displacement quiver over material subdomains")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close()


Ve = VectorElement("CG", mesh.ufl_cell(), 2)
Re = VectorElement("R", mesh.ufl_cell(), 0)
W = FunctionSpace(mesh, MixedElement([Ve, Re]), constrained_domain=PeriodicBoundary(vertices, tolerance=1e-10))
V = FunctionSpace(mesh, Ve)

v_,lamb_ = TestFunctions(W)
dv, dlamb = TrialFunctions(W)
w = Function(W)
dx = Measure('dx')(subdomain_data=subdomains) # see line 20: subdomains = MeshFunction...
#dx(0) will integrate over phase 0 and dx(1) will integrate over phase 1.

Eps = Constant(((0, 0), (0, 0)))
F = sum([inner(sigma(dv, i, Eps), eps(v_))*dx(i) for i in range(nphases)])
a, L = lhs(F), rhs(F)
a += dot(lamb_,dv)*dx + dot(dlamb,v_)*dx


def macro_strain(i):# i is not related to the index of the phase. different i corresponds to different loading
    Eps_Voigt = np.zeros((3,))
    Eps_Voigt[i] = 1
    return np.array([[Eps_Voigt[0], Eps_Voigt[2]/2.], 
                    [Eps_Voigt[2]/2., Eps_Voigt[1]]])
def stress2Voigt(s):
    return as_vector([s[0,0], s[1,1], s[0,1]])

Chom = np.zeros((3, 3))
for (j, case) in enumerate(["Exx", "Eyy", "Exy"]):
    print("Solving {} case...".format(case))
    Eps.assign(Constant(macro_strain(j)))
    solve(a == L, w, [], solver_parameters={"linear_solver": "cg"})
    (v, lamb) = split(w)
    v_fun, lamb_fun = w.split(deepcopy=True)
    #vector plot
    save_displacement_quiver( mesh, subdomains, v_fun, Eps_np = macro_strain(j), 
    filename="quiver_displacement_{}.png".format(case), scale_factor=0.5, max_arrows=600)
    #quiver
    y = SpatialCoordinate(mesh)
    plt.figure()
    p = plot(0.5*(dot(Eps, y)+v), mode="displacement", title=case)
    plt.savefig("displacement_{}.png".format(case), dpi=300)
    Sigma = np.zeros((3,))
    for k in range(3):  
        Sigma[k] = assemble(sum([stress2Voigt(sigma(v, i, Eps))[k]*dx(i) for i in range(nphases)]))/vol
    Chom[j, :] = Sigma

print(np.array_str(Chom, precision=2))

## plotting deformed unit cell with total displacement u = Eps*y + v
#y = SpatialCoordinate(mesh)
#plt.figure()
#p = plot(0.5*(dot(Eps, y)+v), mode="displacement", title=case)
#plt.savefig("displacement.png", dpi=300)
plt.close()