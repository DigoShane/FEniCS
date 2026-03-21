import dolfin 
#print(dolfin.__version__)
from dolfin import *
from dolfinx import *
import matplotlib.pyplot as plt
import fenics as fe
from ufl import tanh

#parameters['allow_extrapolation'] = True

height = 1.0
width = 1.0
length = 1.0
mesh = BoxMesh(Point(0.0, 0.0, 0.0), Point(length,width,height), 12, 12, 12)

## Create submesh ##

marker = MeshFunction("size_t", mesh, mesh.topology().dim() - 1, 0)
for f in facets(mesh):
    marker[f] = height - DOLFIN_EPS < f.midpoint().z() < height + DOLFIN_EPS
    
submesh = MeshView.create(marker, 1)

V = VectorFunctionSpace(mesh, 'CG', 2)

X = FiniteElement('CG', submesh.ufl_cell(), 2)
Y = FiniteElement('CG', submesh.ufl_cell(), 2)
Z = X * Y
S = FunctionSpace(submesh, Z)

T = FunctionSpace(mesh, 'CG', 2)

v = Function(V)

S_1 = Function(S)
S_11, S_12 = S_1.split()

T_S_1 = interpolate(S_11, T)
T_S_2 = interpolate(S_12, T)
#next line results in an error
assigner = FunctionAssigner(V, [T, T])
assigner.assign(v, [T_S_1, T_S_2])
