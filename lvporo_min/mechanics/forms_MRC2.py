from dolfin import *
import sys

from ufl import indices
import dolfin as dolfin
from ufl import indices
from ufl import cofac


class Forms(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        self.parameters.update(params)

        Matmodel = self.parameters["material model"]["Name"]
        assert (
            Matmodel == "Guccione" or Matmodel == "HolzapfelOgden"
        ), "Material model not implemented"

        self.parameters.update({"F": self.Fe()})
        if Matmodel == "Guccione":
            from .GuccionePas import GuccionePas as Passive

        if Matmodel == "HolzapfelOgden":
            from .holzapfelogden import HolzapfelOgden as Passive

        self.passiveforms = Passive(self.parameters)
        self.matparams = self.passiveforms.Getmatparam()

    def default_parameters(self):
        return {
            "material model": {"Name": "Guccione"},
            "Kappa": 1e5,
            "incompressible": True,
        }

    def PassiveMatSEF(self):
        Wp = self.passiveforms.PassiveMatSEF() + self.Wvolumetric()
        return Wp

    def PassiveRubSEF(self):
        Wp = self.passiveforms.PassiveRubSEF() + self.Wvolumetric()
        return Wp

    def PassiveAortaSEF(self):
        Wp = self.passiveforms.PassiveAortaSEF() + self.Wvolumetric()
        return Wp

    def PK1(self):
        PK1 = self.passiveforms.PK1() + self.PK1volumetric()
        return PK1

    def PK2(self):
        PK2 = inv(self.Fmat()) * self.PK1()
        return PK2

    def sigma(self):
        sigma = 1.0 / self.J() * self.PK1() * transpose(self.Fmat())
        return sigma

    def Fmat(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        return F

    def Fe(self):
        Fg = self.parameters["growth_tensor"]
        F = self.Fmat()
        i, j, k = indices(3)
        if Fg is None:
            Fe = F
        else:
            Fe = as_tensor(F[i, j] * inv(Fg)[j, k], (i, k))

        return Fe

    def Emat(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = self.Fe()
        i, j, k = indices(3)
        return 0.5 * (as_tensor(F[k, i] * F[k, j] - I[i, j], (i, j)))

    def J(self):
        F = self.Fe()
        return det(F)

    def Wvolumetric(self):
        isincomp = self.parameters["incompressible"]
        u = self.parameters["displacement_variable"]
        d = u.geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        F = dolfin.variable(F)
        J = det(F)

        if isincomp:
            p = self.parameters["pressure_variable"]
            Wvolumetric = -1.0 * p * (J - 1.0)
        else:
            Kappa = self.parameters["Kappa"]
            Wvolumetric = Kappa / 2.0 * (J - 1.0) ** 2.0

        return Wvolumetric

    def PK1volumetric(self):
        isincomp = self.parameters["incompressible"]
        u = self.parameters["displacement_variable"]
        d = u.geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        F = dolfin.variable(F)
        J = det(F)

        if isincomp:
            p = self.parameters["pressure_variable"]
            Wvolumetric = -1.0 * p * (J - 1.0)
            PK1volumetric = dolfin.diff(Wvolumetric, F)
        else:
            Kappa = self.parameters["Kappa"]
            Wvolumetric = Kappa / 2.0 * (J - 1.0) ** 2.0
            PK1volumetric = dolfin.diff(Wvolumetric, F)

        return PK1volumetric

    def LVcavityvol_mvb(self):  # cavity volume for lv with moving base
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )

        ds_ = Measure(
            "ds",
            domain=mesh,
            subdomain_data=self.parameters["facetboundaries"],
            subdomain_id=self.parameters["topid"],
            metadata={"quadrature_degree": 4},
        )

        F = self.Fmat()
        # vol_form = (
        #    -Constant(1.0 / 3.0)
        #    * inner(det(F) * dot(inv(F).T, N), X + u)
        #    * (ds(self.parameters["LVendoid"]) + ds(2) + ds(3))
        # )

        area = assemble(1.0 * ds_)
        vol_x = assemble((X[0] + u[0]) * ds(self.parameters["topid"])) / area
        vol_y = assemble((X[1] + u[1]) * ds(self.parameters["topid"])) / area
        vol_z = assemble((X[2] + u[2]) * ds(self.parameters["topid"])) / area
        b = Constant((vol_x, vol_y, vol_z))

        vol_form = (
            -Constant(1.0 / 3.0)
            * inner(det(F) * dot(inv(F).T, N), X + u - b)
            * ds(self.parameters["LVendoid"])
        )

        return assemble(vol_form, form_compiler_parameters={"representation": "uflacs"})

    def LVcavityvol_waorta(self):  # cavity volume for lv with aorta
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )

        F = self.Fmat()
        vol_form = (
            -Constant(1.0 / 3.0)
            * inner(det(F) * dot(inv(F).T, N), X + u)
            * (
                ds(self.parameters["LVendoid"])
                + ds(self.parameters["aortic_vplane"])
                + ds(self.parameters["mitral_vplane"])
            )
        )

        return assemble(vol_form, form_compiler_parameters={"representation": "uflacs"})

    def LVcavityvol_fch(self):  # cavity volume for lv in fch mesh
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )

        F = self.Fmat()
        vol_form = (
            -Constant(1.0 / 3.0)
            * inner(det(F) * dot(inv(F).T, N), X + u)
            * (
                ds(self.parameters["LVendoid"])
                + ds(self.parameters["aortic_vplane"])
                + ds(self.parameters["mitral_vplane"])
            )
        )

        return assemble(vol_form, form_compiler_parameters={"representation": "uflacs"})

    def RVcavityvol_fch(self):  # cavity volume for rv in fch mesh
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )

        F = self.Fmat()
        vol_form = (
            -Constant(1.0 / 3.0)
            * inner(det(F) * dot(inv(F).T, N), X + u)
            * (
                ds(self.parameters["RVendoid"])
                + ds(self.parameters["first_rv_valve"])
                + ds(self.parameters["second_rv_valve"])
            )
        )

        return assemble(vol_form, form_compiler_parameters={"representation": "uflacs"})

    def topspringbc(self):  # for v_base computation
        N = self.parameters["facet_normal"]
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )
        pe = self.parameters["lv_constrained_pres"]
        F = self.Fmat()
        J = self.J()

        # JFN = det(F) * dot(inv(F).T, N)

        JFN = J * inv(F.T) * N
        JFN_norm = sqrt(dot(JFN, JFN))

        JFN_x = assemble(pe * JFN[0] * ds(self.parameters["LVendoid"]))
        JFN_y = assemble(pe * JFN[1] * ds(self.parameters["LVendoid"]))
        JFN_z = assemble(pe * JFN[2] * ds(self.parameters["LVendoid"]))

        JFN_all = as_vector((JFN_x, JFN_y, JFN_z))
        # JFN_all = p * J * inv(F.T) * N * ds(self.parameters["LVendoid"])
        # JFN_norm = sqrt(dot(JFN, JFN))

        int_JFN = assemble(JFN_norm * ds(self.parameters["topid"]))

        return JFN_norm / int_JFN, JFN_all

    def LVcavityvol(self):
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )

        F = self.Fmat()

        if isinstance(self.parameters["LVendoid"], list):
            cnt = 0
            for id_ in self.parameters["LVendoid"]:
                if cnt == 0:
                    surface_ = ds(id_)
                else:
                    surface_ += ds(id_)

                cnt += 1
        else:
            surface_ = ds(self.parameters["LVendoid"])

        # vol_form = vol_form_ * ( surface_ )

        vol_form = (
            -Constant(1.0 / 3.0)
            * inner(det(F) * dot(inv(F).T, N), X + u)
            * surface_  # ds(self.parameters["LVendoid"])
        )

        return assemble(vol_form, form_compiler_parameters={"representation": "uflacs"})

    def RVcavityvol(self):
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        X = SpatialCoordinate(mesh)
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )

        F = self.Fmat()

        if isinstance(self.parameters["RVendoid"], list):
            cnt = 0
            for id_ in self.parameters["RVendoid"]:
                if cnt == 0:
                    surface_ = ds(id_)
                else:
                    surface_ += ds(id_)

                cnt += 1
        else:
            surface_ = ds(self.parameters["RVendoid"])

        vol_form = (
            -Constant(1.0 / 3.0)
            * inner(det(F) * dot(inv(F).T, N), X + u)
            * surface_  # ds(self.parameters["RVendoid"])
        )

        return assemble(vol_form, form_compiler_parameters={"representation": "uflacs"})

    def LVcavitypressure(self):
        W = self.parameters["mixedfunctionspace"]
        w = self.parameters["mixedfunction"]
        mesh = self.parameters["mesh"]

        comm = W.mesh().mpi_comm()
        dofmap = W.sub(self.parameters["LVendo_comp"]).dofmap()
        val_dof = dofmap.cell_dofs(0)[0]

        # the owner of the dof broadcasts the value
        own_range = dofmap.ownership_range()

        try:
            val_local = w.vector()[val_dof]
        except IndexError:
            val_local = 0.0

        pressure = MPI.sum(comm, val_local)

        return pressure

    def LVcavitypres(self):
        pe = self.parameters["lv_constrained_pres"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )

        J = self.J()
        u = self.parameters["displacement_variable"]
        F = self.Fmat()

        if isinstance(self.parameters["LVPid"], list):
            cnt = 0
            for id_ in self.parameters["LVPid"]:
                if cnt == 0:
                    surface_ = ds(id_)
                else:
                    surface_ += ds(id_)

                cnt += 1
        else:
            surface_ = ds(self.parameters["LVPid"])

        pres = (
            pe * inner(J * inv(F.T) * N, u) * surface_
        )  # ds(self.parameters["LVendoid"])

        # pres = 1 * dsendo
        return pres

    def RVcavitypres(self):
        pe = self.parameters["rv_constrained_pres"]
        N = self.parameters["facet_normal"]
        mesh = self.parameters["mesh"]
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )

        J = self.J()
        u = self.parameters["displacement_variable"]
        F = self.Fmat()
        # dsendo = ds(
        #    self.parameters["LVendoid"],
        #    domain=self.parameters["mesh"],
        #    subdomain_data=self.parameters["facetboundaries"],
        # )

        if not ("RVPid" in list(self.parameters.keys())):
            self.parameters.update({"RVPid": self.parameters["RVendoid"]})

        if isinstance(self.parameters["RVPid"], list):
            cnt = 0
            for id_ in self.parameters["RVPid"]:
                if cnt == 0:
                    surface_ = ds(id_)
                else:
                    surface_ += ds(id_)

                cnt += 1
        else:
            surface_ = ds(self.parameters["RVPid"])

        pres = (
            pe * inner(J * inv(F.T) * N, u) * surface_
        )  # ds(self.parameters["RVendoid"])

        # pres = 1 * dsendo
        return pres

    def RVcavitypressure(self):
        W = self.parameters["mixedfunctionspace"]
        w = self.parameters["mixedfunction"]
        mesh = self.parameters["mesh"]

        comm = W.mesh().mpi_comm()
        dofmap = W.sub(self.parameters["RVendo_comp"]).dofmap()
        val_dof = dofmap.cell_dofs(0)[0]

        # the owner of the dof broadcasts the value
        own_range = dofmap.ownership_range()

        try:
            val_local = w.vector().get_local()[val_dof]
        except IndexError:
            val_local = 0.0

        pressure = MPI.sum(comm, val_local)

        return pressure

    def LVV0constrainedE(self):
        mesh = self.parameters["mesh"]
        u = self.parameters["displacement_variable"]
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )
        dsendo = ds(
            self.parameters["LVendoid"],
            domain=self.parameters["mesh"],
            subdomain_data=self.parameters["facetboundaries"],
        )
        area = self.parameters["LVendo_area"]
        pendo = self.parameters["lv_volconst_variable"]
        V0 = self.parameters["lv_constrained_vol"]

        X = SpatialCoordinate(mesh)
        x = u + X

        F = self.Fmat()
        N = self.parameters["facet_normal"]
        n = cofac(F) * N

        V_u = -Constant(1.0 / 3.0) * inner(x, n)
        Wvol = (Constant(1.0) / area * pendo * V0 * dsendo) - (pendo * V_u * dsendo)
        #        Wvol = (Constant(1.0)/area  * V0 * dsendo) - (V_u *dsendo)
        return Wvol

    def RVV0constrainedE(self):
        mesh = self.parameters["mesh"]
        u = self.parameters["displacement_variable"]
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )
        dsendo = ds(
            self.parameters["RVendoid"],
            domain=self.parameters["mesh"],
            subdomain_data=self.parameters["facetboundaries"],
        )
        pendo = self.parameters["rv_volconst_variable"]
        V0 = self.parameters["rv_constrained_vol"]

        X = SpatialCoordinate(mesh)
        x = u + X

        F = self.Fmat()
        N = self.parameters["facet_normal"]
        n = cofac(F) * N

        area = assemble(
            Constant(1.0) * dsendo,
            form_compiler_parameters={"representation": "uflacs"},
        )
        V_u = -Constant(1.0 / 3.0) * inner(x, n)
        Wvol = (Constant(1.0 / area) * pendo * V0 * dsendo) - (pendo * V_u * dsendo)

        return Wvol

    def fiberstress(self):
        F = self.Fmat()
        J = self.J()
        PK1 = self.PK1()

        Tca = (1.0 / J) * PK1 * F.T
        Sca = inv(F) * PK1

        f0 = self.parameters["fiber"]
        i, j = indices(2)
        return f0[i] * Sca[i, j] * f0[j]

    def fiberstrain(self, F_ref):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)

        F = self.Fe() * inv(F_ref)
        f0 = self.parameters["fiber"]
        i, j, k = indices(3)
        Emat = 0.5 * (as_tensor(F[k, i] * F[k, j] - I[i, j], (i, j)))

        return f0[i] * Emat[i, j] * f0[j]

    def fiberwork(self, F_ref):
        F = self.Fe() * inv(F_ref)

        J = self.J()
        PK1 = self.PK1()

        Tca = (1.0 / J) * PK1 * F.T
        Sca = inv(F) * PK1

        f0 = self.parameters["fiber"]
        Emat = self.Emat()

        i, j = indices(2)
        f = f0[i] * Sca[i, j] * f0[j]
        s = f0[i] * Emat[i, j] * f0[j]

        return dot(f, s)

    def solveLaplaceEquation_fch(self):
        df_mesh = self.parameters["mesh"]
        df_facet = self.parameters["facetboundaries"]

        V = FunctionSpace(df_mesh, FiniteElement("Lagrange", df_mesh.ufl_cell(), 1))
        Vvec = FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0))

        bas_ids = [self.parameters["aorta_wall"], self.parameters["pulm_wall"]]
        bc_bas = [DirichletBC(V, Constant(0.25), df_facet, id_) for id_ in bas_ids]

        bc_apx = [DirichletBC(V, Constant(1.0), df_facet, self.parameters["apxid"])]

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(0)
        a = inner(nabla_grad(u), nabla_grad(v)) * dx
        L = f * v * dx

        # Compute solution
        u = Function(V)
        solve(
            a == L, u, bc_bas + bc_apx
        )  # , solver_parameters={"linear_solver": "petsc"})

        grad_u = project(grad(u), Vvec)  # , solver_type="petsc")

        return u  # , grad_u

    def solveLaplaceEquation_waorta(self):
        df_mesh = self.parameters["mesh"]
        df_facet = self.parameters["facetboundaries"]

        V = FunctionSpace(df_mesh, FiniteElement("Lagrange", df_mesh.ufl_cell(), 1))
        Vvec = FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0))

        bas_ids = [
            self.parameters["aorta_int_wall"],
            self.parameters["aorta_ext_wall"],
            self.parameters["aorta_ring"],
        ]
        bc_bas = [DirichletBC(V, Constant(0.25), df_facet, id_) for id_ in bas_ids]
        bc_apx = [DirichletBC(V, Constant(1.0), df_facet, self.parameters["apxid"])]

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(0)
        a = inner(nabla_grad(u), nabla_grad(v)) * dx
        L = f * v * dx

        # Compute solution
        u = Function(V)
        solve(
            a == L, u, bc_bas + bc_apx
        )  # , solver_parameters={"linear_solver": "petsc"})

        grad_u = project(grad(u), Vvec)  # , solver_type="petsc")

        return u  # , grad_u

    def solveLaplaceEquation_ideal(self):
        df_mesh = self.parameters["mesh"]
        df_facet = self.parameters["facetboundaries"]

        V = FunctionSpace(df_mesh, FiniteElement("Lagrange", df_mesh.ufl_cell(), 1))
        Vvec = FunctionSpace(df_mesh, VectorElement("DG", df_mesh.ufl_cell(), 0))

        bc_bas = [DirichletBC(V, Constant(0.5), df_facet, self.parameters["topid"])]
        bc_apx = [DirichletBC(V, Constant(2.0), df_facet, self.parameters["apxid"])]

        # Define variational problem
        u = TrialFunction(V)
        v = TestFunction(V)
        f = Constant(0)
        a = inner(nabla_grad(u), nabla_grad(v)) * dx
        L = f * v * dx

        # Compute solution
        u = Function(V)
        solve(
            a == L, u, bc_bas + bc_apx
        )  # , solver_parameters={"linear_solver": "petsc"})

        grad_u = project(grad(u), Vvec)  # , solver_type="petsc")

        return u  # , grad_u

    def IMP(self):
        u = self.parameters["displacement_variable"]
        J = self.J()
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]

        # F = self.Fmat()
        F = self.Fe()
        PK1 = self.PK1()

        Tca = (1.0 / J) * PK1 * F.T

        s = F * s0 / sqrt(inner(F * s0, F * s0))
        n = F * n0 / sqrt(inner(F * n0, F * n0))

        i, j = indices(2)
        Ipressure = s[i] * Tca[i, j] * s[j]

        return Ipressure

    def IMP2(self):
        u = self.parameters["displacement_variable"]
        J = self.J()
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]

        # F = self.Fmat()
        F = self.Fe()
        PK1 = self.PK1()

        Tca = (1.0 / J) * PK1 * F.T

        s = F * s0 / sqrt(inner(F * s0, F * s0))
        n = F * n0 / sqrt(inner(F * n0, F * n0))

        i, j = indices(2)
        Ipressure = 0.5 * (s[i] * Tca[i, j] * s[j] + n[i] * Tca[i, j] * n[j])

        return Ipressure

    def IMPendo(self):
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        F = self.Fe()
        PK1 = self.PK1()

        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )
        n = F * N / sqrt(inner(F * N, F * N))

        i, j = indices(2)
        Ipressure = -n[i] * PK1[i, j] * N[j] * ds(self.parameters["LVendoid"])

        return Ipressure

    def IMPepi(self):
        u = self.parameters["displacement_variable"]
        N = self.parameters["facet_normal"]
        F = self.Fe()
        PK1 = self.PK1()

        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )
        n = F * N / sqrt(inner(F * N, F * N))

        i, j = indices(2)
        Ipressure = -n[i] * PK1[i, j] * N[j] * ds(self.parameters["epiid"])

        return Ipressure

    def areaendo(self):
        u = self.parameters["displacement_variable"]
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]

        d = u.geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        J = det(F)

        s = F * s0 / sqrt(inner(F * s0, F * s0))

        N = self.parameters["facet_normal"]
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )
        n = F * N / sqrt(inner(F * N, F * N))

        i, j = indices(2)
        return (J * inv(F.T) * N)[i] * n[i] * ds(self.parameters["endoid"])

    def areaepi(self):
        u = self.parameters["displacement_variable"]
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]

        d = u.geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        J = det(F)

        s = F * s0 / sqrt(inner(F * s0, F * s0))

        N = self.parameters["facet_normal"]
        ds = dolfin.ds(
            subdomain_data=self.parameters["facetboundaries"],
            metadata={"quadrature_degree": 4},
        )
        n = F * N / sqrt(inner(F * N, F * N))

        i, j = indices(2)
        return (J * inv(F.T) * N)[i] * n[i] * ds(self.parameters["epiid"])

    def poro_PK_1(self):
        PK_1 = (
            self.passiveforms.PK1()
            + self.passiveforms.poro_volumetricstress()
            #+ self.passiveforms.poro_porositystress()
        )
        return PK_1
    
    def poro_fibermat(self):
        Mat,I = self.passiveforms.fibermat()
        return Mat,I

    # def poro_pressure(self):
    #     pressure_ = self.passiveforms.poro_pressure()
    #     return pressure_
