import sys

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from contextlib import ExitStack
from petsc4py.PETSc import ScalarType

import ufl
import dolfinx.io
import dolfinx.fem as fem
import dolfinx.mesh

import meshio #pip install meshio[all]


def get_xy_bounding_box(x: np.ndarray):
    """Compute a 2D bounding box for a given set of points.
    Args:
        x (np.ndarray): set of points
    Returns:
        (2, 2) np.ndarray: bounding box
    """
    x_min = np.min(x[:, 0])
    y_min = np.min(x[:, 1])
    x_max = np.max(x[:, 0])
    y_max = np.max(x[:, 1])
    return np.array([[x_min, y_min], [x_max, y_max]])

def stiffness_tensor_isotropic_ufl(E, nu):
    return (
        E
        / ((1 + nu) * (1 - 2 * nu))
        * ufl.as_matrix(
            [
                [1 - nu, nu, nu, 0, 0, 0],
                [nu, 1 - nu, nu, 0, 0, 0],
                [nu, nu, 1 - nu, 0, 0, 0],
                [0, 0, 0, 1 - 2 * nu, 0, 0],
                [0, 0, 0, 0, 1 - 2 * nu, 0],
                [0, 0, 0, 0, 0, 1 - 2 * nu],
            ]
        )
    )


def strain2voigt(e):
    """e is a 2nd-order tensor, returns its Voigt vectorial representation"""
    return ufl.as_vector([e[0, 0], e[1, 1], e[2, 2], 2 * e[1, 2], 2 * e[0, 2], 2 * e[0, 1]])

def stress2voigt(s):
    """e is a 2nd-order tensor, returns its Voigt vectorial representation"""
    return ufl.as_vector([s[0, 0], s[1, 1], s[2, 2], s[1, 2], s[0, 2], s[0, 1]])

def voigt2stress(s):
    return ufl.as_tensor([[s[0], s[5], s[4]], [s[5], s[1], s[3]], [s[4], s[3], s[2]]])

def ufl_strain(v):
    return ufl.sym(ufl.grad(v))

def ufl_stress(v, stiffness_tensor):
    return voigt2stress(ufl.dot(stiffness_tensor, strain2voigt(ufl_strain(v))))

def sigma_VM(v, stiffness_tensor):
    sigma = ufl_stress(v, stiffness_tensor)
    deviatoric = sigma - 1./3. * ufl.tr(sigma) * ufl.Identity(len(v))
    return ufl.sqrt( 2./3. * ufl.inner(deviatoric,deviatoric) )

def read_in_mesh(str):

    def create_mesh(mesh, cell_type, prune_z=True):
        cells = mesh.get_cells_type(cell_type)
        points = mesh.points[:,:2] if prune_z else mesh.points
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells})
        return out_mesh
    msh = meshio.read(str)
    triangle_mesh = create_mesh(msh, "tetra", prune_z=False)
    meshio.write("mesh/mesh.xdmf", triangle_mesh)
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "mesh/mesh.xdmf", "r") as xdmf:
        mesh = xdmf.read_mesh(name="Grid")
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim-1)

    return mesh

if __name__ == "__main__":

    print("sys.argv",sys.argv)
    if len(sys.argv)>1:
        n_simu = int(sys.argv[1])
    else:
        n_simu = 1 

    mesh = read_in_mesh("mesh/Dogbone_hole_refined_non_sym.msh")

    bb = get_xy_bounding_box(mesh.geometry.x)
    boundary_left = lambda x: x[0]<=bb[0,0]+1.e-10
    boundary_right = lambda x: x[0]>=bb[1,0]-1.e-10

    facets_left = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim - 1, boundary_left)
    facets_right = dolfinx.mesh.locate_entities(mesh, mesh.topology.dim - 1, boundary_right)
    facets = np.hstack([facets_left,facets_right])
    markers_left = np.full_like(facets_left, 1)
    markers_right = np.full_like(facets_right, 2)
    markers = np.hstack([markers_left,markers_right])
    facet_tag = dolfinx.mesh.meshtags(mesh, mesh.topology.dim - 1, facets, markers)
    ds = ufl.Measure("ds", domain=mesh, subdomain_data=facet_tag)

    mesh.topology.create_connectivity(mesh.topology.dim-1, mesh.topology.dim)
    with dolfinx.io.XDMFFile(mesh.comm, "out/facet_tags.xdmf", "w") as xdmf:
        xdmf.write_mesh(mesh)
        xdmf.write_meshtags(facet_tag)

    def dirichlet_bcs(mesh, V, disp_amp):
        
        disp_value = np.array((disp_amp,0.,0.) , dtype=PETSc.ScalarType) 

        facet_dim = mesh.topology.dim - 1
        boundary_left_facets = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, boundary_left)
        boundary_right_facets = dolfinx.mesh.locate_entities_boundary(mesh, facet_dim, boundary_right)

        boundary_left_dofs = fem.locate_dofs_topological(V, facet_dim, boundary_left_facets)
        boundary_right_dofs = fem.locate_dofs_topological(V, facet_dim, boundary_right_facets)

        bc_left = fem.dirichletbc( fem.Constant(mesh, -disp_value) , boundary_left_dofs, V)
        bc_right = fem.dirichletbc( fem.Constant(mesh, disp_value) , boundary_right_dofs, V)

        bcs = [bc_left, bc_right]
        return bcs

    # define function spaces and variational forms
    CG1 = fem.FunctionSpace(mesh, ("CG", 1))
    CG1_vector = fem.VectorFunctionSpace(mesh, ("CG", 1))
    DG1_vector = fem.VectorFunctionSpace(mesh, ("DG", 1))

    DG0 = fem.FunctionSpace(mesh, ("DG", 0))
    DG0_tensor = fem.TensorFunctionSpace(mesh, ("DG", 0), shape=(mesh.topology.dim, mesh.topology.dim))

    def coeff_CIP(unif,mesh_size):
        one_minus_proba_drop = 0.9
        factor = 1.
        return factor * 0.5 * ( ufl.tanh( ( unif-one_minus_proba_drop ) / (1.e-8) ) + 1. )

    rand = fem.Function(DG0)
    #print(rand.x.array.shape)
    #rand.x.array[:] = np.random.rand(len(rand.x.array))
    #strain_expression = fem.Expression(ufl_strain(u_sol), DG0_tensor.element.interpolation_points
    rand.x.array[:] = np.zeros_like(rand.x.array) # to generate an unperturbed solution before going into realisation loop

    nitsche_bc = 0
    if nitsche_bc==1:
        bcs = [] 
    else:
        bcs = dirichlet_bcs(mesh, CG1_vector, 1.0)  
    
    u = ufl.TrialFunction(CG1_vector)
    v = ufl.TestFunction(CG1_vector)

    E = 1.0
    #if ind==0:
    #    E = ufl.exp(5.0*0.65) # homogeneous case
    C = stiffness_tensor_isotropic_ufl( E , nu=0.3)

    a = ufl.inner(ufl_stress(u, C), ufl_strain(v)) * ufl.dx

    n = ufl.FacetNormal(mesh)
    h = 2 * ufl.Circumradius(mesh)  

    # n has two values, one for +, one for -
    if 0:
        a -= ufl.inner( ufl.avg( ufl.dot( ufl_stress(v, C) , n) ), ufl.jump(u) ) *ufl.dS 
        a -= ufl.inner( ufl.avg( ufl.dot( ufl_stress(u, C) , n) ), ufl.jump(v) ) *ufl.dS
        a += 1.0e2*E/ufl.avg(h) *ufl.dot( ufl.jump(u),ufl.jump(v) ) *ufl.dS   
    else:      
        a += 1.0e-1*E*coeff_CIP(rand("+"),ufl.avg(h))*ufl.avg(h)*ufl.dot( ufl.jump( ufl.grad(u) , n ), ufl.jump( ufl.grad(v) , n ) ) *ufl.dS   

    if nitsche_bc==1:
        a -= ufl.inner( ufl.dot( ufl_stress(v, C) , n) , u ) *(ds(1)+ds(2))
        a -= ufl.inner( ufl.dot( ufl_stress(u, C) , n) , v ) *(ds(1)+ds(2))
        a += 1.0e1*E/h *ufl.dot( u , v ) *(ds(1)+ds(2))

    #F -= ufl.dot(ufl.avg(ufl.grad(v)), ufl.jump(u,n)) *ufl.dS 
    #F -= ufl.dot(ufl.avg(ufl.grad(u)), ufl.jump(v,n)) *ufl.dS
    #F += 1.0e1*1./ufl.avg(h) *ufl.jump(u)*ufl.jump(v) *ufl.dS 

    u_d = fem.Constant(mesh, ScalarType((1.0, 0., 0.)))
    f_d = fem.Constant(mesh, ScalarType((0, 0., 0)))

    #L = ufl.dot(fem.Constant(mesh, [0., 1., 0.]), v) * ufl.dx
    L = ufl.dot( f_d , v ) * ufl.dx
    if nitsche_bc==1:
        L -= ufl.inner( ufl.dot( ufl_stress(v, C) , n) , u_d ) * ds(2)
        L += 1.0e1*1./h *ufl.dot( u_d , v ) * ds(2)

    problem = fem.petsc.LinearProblem(a, L, bcs)

    u_sol_ref = problem.solve()
    u_sol_ref.name = "u"

    # compute norm of solution
    u_norm_expr = ufl.dot(u_sol_ref, u_sol_ref) * ufl.dx
    u_norm = fem.assemble_scalar(fem.form(u_norm_expr))
    print(f"u norm: {u_norm}")

    # apply dirichlet boundary condition
    #disp_value = 1.0
    #u_left = fem.Constant(mesh, -disp_value)
    #u_right = fem.Constant(mesh, disp_value)
    #u_bottom = fem.Constant(mesh, -disp_value)
    #u_top = fem.Constant(mesh, disp_value)

    #boundary_left_dofs = fem.locate_dofs_topological(DG1_vector.sub(0), facet_dim, boundary_left_facets)
    #boundary_right_dofs = fem.locate_dofs_topological(DG1_vector.sub(0), facet_dim, boundary_right_facets)
    #boundary_bottom_dofs = fem.locate_dofs_topological(DG1_vector.sub(1), facet_dim, boundary_bottom_facets)
    #boundary_top_dofs = fem.locate_dofs_topological(DG1_vector.sub(1), facet_dim, boundary_top_facets)
    #boundary_left_dofs = fem.locate_dofs_geometrical(DG1_vector, boundary_left)
    #boundary_right_dofs = fem.locate_dofs_geometrical(DG1_vector, boundary_right)
    #boundary_bottom_dofs = fem.locate_dofs_geometrical(DG1_vector, boundary_bottom)
    #boundary_top_dofs = fem.locate_dofs_geometrical(DG1_vector, boundary_top)

    #bc_left = fem.dirichletbc(u_left, boundary_left_dofs, DG1_vector.sub(0))
    #bc_right = fem.dirichletbc(u_right, boundary_right_dofs, DG1_vector.sub(0))
    #bc_bottom = fem.dirichletbc(u_bottom, boundary_bottom_dofs, DG1_vector.sub(1))
    #bc_top = fem.dirichletbc(u_top, boundary_top_dofs, DG1_vector.sub(1))

    fic = dolfinx.io.XDMFFile(mesh.comm, "out/res.xdmf", "w")
    fic.write_mesh(mesh)

    for ind_simu in range(n_simu):

        print("realisation --\\ ",ind_simu," //--")

        rand.x.array[:] = np.random.rand(len(rand.x.array))

        coeff_CIP_expr = fem.Expression(coeff_CIP(rand,1.), DG0.element.interpolation_points)
        coeff_CIP_fun = fem.Function(DG0, name="coeff_CIP")
        coeff_CIP_fun.interpolate(coeff_CIP_expr)
        with dolfinx.io.VTKFile(MPI.COMM_WORLD, "out/coeff_CIP.pvd", "w") as file:
            file.write_function(coeff_CIP_fun)

        problem = fem.petsc.LinearProblem(a, L, bcs)

        u_sol = problem.solve()
        u_sol.name = "u"

        # compute norm of solution
        u_norm_expr = ufl.dot(u_sol, u_sol) * ufl.dx
        u_norm = fem.assemble_scalar(fem.form(u_norm_expr))
        print(f"u norm: {u_norm}")
    
        strain_expression = fem.Expression(ufl_strain(u_sol), DG0_tensor.element.interpolation_points)
        stress_expression = fem.Expression(ufl_stress(u_sol, C), DG0_tensor.element.interpolation_points)
        stress_vm_expression = fem.Expression(sigma_VM(u_sol, C), DG0.element.interpolation_points)

        strain_sol = fem.Function(DG0_tensor, name="eps")
        stress_sol = fem.Function(DG0_tensor, name="sigma")
        stress_vm_sol = fem.Function(DG0, name="sigma_vm")
        delta_sol = fem.Function(DG1_vector, name="delta_u")

        strain_sol.interpolate(strain_expression)
        stress_sol.interpolate(stress_expression)
        stress_vm_sol.interpolate(stress_vm_expression)

        delta = fem.Expression(u_sol - u_sol_ref, DG1_vector.element.interpolation_points)
        delta_sol.interpolate(delta)

        # export fields to XDMF file
        #with dolfinx.io.XDMFFile(MPI.COMM_WORLD, "out/solution"+str(ind_simu)+".xdmf", "w") as file:
        #    file.write_mesh(mesh)
        #    file.write_function(u_sol)
        #    file.write_function(strain_sol)
        #    file.write_function(stress_sol)
        #    file.write_function(stress_vm_sol)
        fic.write_function(u_sol, ind_simu) 
        fic.write_function(strain_sol, ind_simu) 
        fic.write_function(stress_sol, ind_simu) 
        fic.write_function(stress_vm_sol, ind_simu)
        fic.write_function(delta_sol, ind_simu) 

fic.close()

#    with dolfinx.io.VTKFile(MPI.COMM_WORLD, "out/u.pvd", "w") as file:
#        file.write_function(u_sol)

#    with dolfinx.io.VTKFile(MPI.COMM_WORLD, "out/rand_prop.pvd", "w") as file:
#        file.write_function(prop_dolfinx)    

    #with dolfinx.io.VTKFile(MPI.COMM_WORLD, "out/vm.pvd", "w") as file:
    #    file.write_function(stress_vm_CG1_sol)    

    #else:
    #    file = dolfinx.io.VTKFile(MPI.COMM_WORLD, "out/stress_bulk0.pvd", "w") 
    #    file.write([stress2voigt_bulk_sol._cpp_object])
    #    file = dolfinx.io.VTKFile(MPI.COMM_WORLD, "out/stress_shear0.pvd", "w") 
    #    file.write([stress2voigt_shear_sol._cpp_object])
