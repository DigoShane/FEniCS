from dolfin import *
import fenics as fe
import numpy as np
import matplotlib.pyplot as plt

Lx = 20
mesh = fe.IntervalMesh(500, 0, Lx)
V = FunctionSpace(mesh, "CG", 2)

Hin = float(1 / np.sqrt(2))
rlx_par_in = 0.5
tol_abs_in = 1E-6

theta = interpolate(Constant(0.0), V)
plot(theta)
plt.title(
    r"$\theta(x)$ for domain [0," + str(Lx) + "] for H= " + str(Hin)
    + ", with rlx_par " + str(rlx_par_in) + " and abs_tol " + str(tol_abs_in),
    fontsize=26,
)
plt.show()
