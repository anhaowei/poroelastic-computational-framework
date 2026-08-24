
from dolfin import *
import numpy as np

from ..utils.nsolver import NSolver
from ..utils.oops_objects_MRC2 import lv_mesh as lv_mechanics_mesh
from .forms_MRC2 import Forms
from .activeforms_MRC2 import activeForms
from ufl import indices


class MEmodel(object):
    """
    Minimal LV-only mechanics model for the poroelastic closed-loop case.

    Design choices:
    - Supports LV only: SimDet["isLV"] must be True; BiV/FCH/aorta branches are removed.
    - Supports pressure-control mechanics: SimDet["ispctrl"] should be True.
    - Uses a poroelastic residual only: Darcy/mass balance + poroelastic momentum balance
      + active stress + rigid-body constraints.
    - Keeps the public methods used by run_BiV_ClosedLoop_pctrl_lvporo.py so the driver
      can run with only import cleanup.
    """

    def __init__(self, params, SimDet):
        self.parameters = self.default_parameters()
        self.parameters.update(params)
        self.SimDet = SimDet

        self.isLV = bool(SimDet.get("isLV", True))
        self.iswaorta = False
        self.isFCH = False
        self.isBiV = False
        if not self.isLV:
            raise NotImplementedError("This minimal MEmodel3.py supports LV case only.")

        self.ispctrl = bool(SimDet.get("ispctrl", True))
        if not self.ispctrl:
            raise NotImplementedError("This minimal model keeps only pressure-control LV mechanics.")

        self.islumped = bool(SimDet.get("islumped", False))
        self.deg_me = SimDet["GiccioneParams"].get("deg", 4)
        self.discretization = SimDet.get("Mechanics Discretization", "P1P1P0_Berger")
        self.discretization_technique = SimDet.get("Technique Discretization", 1)
        self.use_springbc = bool(SimDet.get("springbc", False))

        # Mechanics mesh: LV only.
        self.Mesh = lv_mechanics_mesh(self.parameters, SimDet)
        self.mesh_me = self.Mesh.mesh
        self.facetboundaries_me = self.Mesh.facetboundaries
        self.edgeboundaries_me = self.Mesh.edgeboundaries
        self.matid_me = self.Mesh.matid
        self.ds_me = self.Mesh.ds
        self.dx_me = self.Mesh.dx

        self.f0_me = self.Mesh.f0
        self.s0_me = self.Mesh.s0
        self.n0_me = self.Mesh.n0

        # Cavity and perfusion pressures are mutable Expressions used by the driver.
        self.LVCavityvol = Expression(("vol"), vol=0.0, degree=2)
        self.RVCavityvol = Expression(("vol"), vol=0.0, degree=2)
        self.LVCavitypres = Expression(("pres"), pres=0.0, degree=2)
        self.RVCavitypres = Expression(("pres"), pres=0.0, degree=2)
        self.p_a = Expression(("coronary"), coronary=0.0, degree=2)
        self.p_v = Expression(("venous"), venous=1300.0, degree=2)

        # LV endocardial area is used by Forms.LVV0constrainedE and retained for compatibility.
        self.LVendo_area_me = Expression(("val"), val=0.0, degree=2)
        self.LVendo_area_me.val = assemble(
            Constant(1.0) * self._surface_measure(SimDet["LVendoid"]),
            form_compiler_parameters={"representation": "uflacs"},
        )

        self._build_function_spaces()
        self.Ftotal, self.Jac, self.bcs = self.Problem()

    def default_parameters(self):
        return {"probeloc": [3.5, 0.0, -2.0]}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _surface_measure(self, marker):
        if isinstance(marker, list):
            m = None
            for mid in marker:
                term = self.ds_me(mid, domain=self.mesh_me, subdomain_data=self.facetboundaries_me)
                m = term if m is None else m + term
            return m
        return self.ds_me(marker, domain=self.mesh_me, subdomain_data=self.facetboundaries_me)

    def _build_function_spaces(self):
        """Berger-type lowest-order three-field spaces.

        Unknown ordering is
            0: displacement u      in [CG1]^3
            1: nominal Darcy flux W in [CG1]^3
            2: pore pressure p      in DG0
            3: optional Real rigid-body multiplier block

        Berger et al. (Comput Mech, 2017) use continuous piecewise-linear
        displacement/fluid-flux fields and piecewise-constant pressure.
        """
        Velem = VectorElement("CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default")
        Zelem = VectorElement("CG", self.mesh_me.ufl_cell(), 1, quad_scheme="default")
        Qelem = FiniteElement("DG", self.mesh_me.ufl_cell(), 0, quad_scheme="default")
        Qelem._quad_scheme = "default"

        if self.use_springbc:
            self.W = FunctionSpace(self.mesh_me, MixedElement([Velem, Zelem, Qelem]))
        else:
            Relem = FiniteElement("Real", self.mesh_me.ufl_cell(), 0, quad_scheme="default")
            Relem._quad_scheme = "default"
            VRelem = MixedElement([Relem, Relem, Relem, Relem, Relem])
            self.W = FunctionSpace(self.mesh_me, MixedElement([Velem, Zelem, Qelem, VRelem]))

        Quadelem = FiniteElement(
            "Quadrature", self.mesh_me.ufl_cell(), degree=self.deg_me, quad_scheme="default"
        )
        Quadelem._quad_scheme = "default"
        Telem2 = TensorElement(
            "Quadrature", self.mesh_me.ufl_cell(), degree=self.deg_me,
            shape=2 * (3,), quad_scheme="default"
        )
        Telem2._quad_scheme = "default"
        for e in Telem2.sub_elements():
            e._quad_scheme = "default"

        self.Quad = FunctionSpace(self.mesh_me, Quadelem)
        self.TF = FunctionSpace(self.mesh_me, Telem2)
        self.Q = FunctionSpace(self.mesh_me, "DG", 0)
        self.QDG = self.Q
        self.QCG1 = FunctionSpace(self.mesh_me, "CG", 1)
        self.V_CG1 = VectorFunctionSpace(self.mesh_me, "CG", 1)

        self.we_n = Function(self.W.sub(0).collapse())
        self.w_me = Function(self.W)
        self.w_me_n = Function(self.W)
        self.dw_me = TrialFunction(self.W)
        self.wtest_me = TestFunction(self.W)

    def _current(self):
        if self.use_springbc:
            u, Wflux, p = self.w_me.split(deepcopy=True)
            return u, Wflux, p, None
        u, Wflux, p, c = self.w_me.split(deepcopy=True)
        return u, Wflux, p, c

    def _previous(self):
        if self.use_springbc:
            u, Wflux, p = self.w_me_n.split(deepcopy=True)
            return u, Wflux, p, None
        u, Wflux, p, c = self.w_me_n.split(deepcopy=True)
        return u, Wflux, p, c

    def _poro_params(self):
        return {
            "permeability": self.SimDet.get("permeability", Constant(0.0)),
            "beta_a": self.SimDet.get("beta_a", Constant(0.0)),
            "beta_v": self.SimDet.get("beta_v", Constant(0.0)),
        }

    def _coronary_arterial_conductance_factor(self):
        """Return the UFL-compatible myocardial-compression conductance factor."""
        model = self.SimDet.get("coronary_compression_model", "none")
        if model == "none":
            return Constant(1.0)
        if model != "plv_sigmoid":
            raise ValueError(
                "Unknown coronary_compression_model: {}".format(model)
            )

        g_min = float(self.SimDet.get("coronary_g_min", 0.42))
        P50 = float(self.SimDet.get("coronary_P50_mmHg", 35.0))
        kP = float(self.SimDet.get("coronary_kP_mmHg", 5.0))
        if not (0.0 < g_min <= 1.0):
            raise ValueError("coronary_g_min must be in (0, 1].")
        if kP <= 0.0:
            raise ValueError("coronary_kP_mmHg must be > 0.")

        P_lv_mmHg = Constant(0.0075) * self.LVCavitypres
        return Constant(g_min) + Constant(1.0 - g_min) / (
            Constant(1.0) + exp((P_lv_mmHg - Constant(P50)) / Constant(kP))
        )

    # ------------------------------------------------------------------
    # Boundary conditions and variational problem
    # ------------------------------------------------------------------
    def set_BCs(self):
        W = self.W
        facetboundaries = self.facetboundaries_me
        bcs = []

        # Original behavior:
        # - without springbc, fix z-displacement on topid and use Real multipliers for remaining modes.
        # - with springbc, do not add this top z DirichletBC; the epicardial spring support stabilizes
        #   the model, while optional fix_surf constraints can still be supplied explicitly.
        if not self.use_springbc:
            topid = self.SimDet["topid"]
            bcs.append(
                DirichletBC(W.sub(0).sub(2), Expression(("0.0"), degree=2), facetboundaries, topid)
            )

        # Optional additional fixed surface(s), retained for compatibility with existing SimDet.
        if "fix_surf" in self.SimDet:
            fix_surf = self.SimDet["fix_surf"]
            markers = fix_surf if isinstance(fix_surf, list) else [fix_surf]
            for marker in markers:
                bcs.append(
                    DirichletBC(
                        W.sub(0), Expression(("0.0", "0.0", "0.0"), degree=2),
                        facetboundaries, marker,
                    )
                )
        return bcs

    def _springbc_residual(self, u_me, u_me_n, v_me, N_me, ds_me):
        """Original LV epicardial spring/dashpot boundary support.

        Normal and tangential spring-dashpot terms are applied on the epicardial marker.
        This mirrors the original LV branch in MEmodel3.py.  The optional moving-base
        compensation is also retained through Forms.topspringbc() when spring_atbase=True.
        """
        dim = u_me.ufl_shape[0]
        I = Identity(dim)
        epiid = self.SimDet["epiid"]
        topid = self.SimDet["topid"]

        k_spring = self.SimDet.get("springparam", [2.0e3, 2.0e3])
        c_damping = self.SimDet.get("dashpotparam", [2.0e2, 2.0e1])
        epiid_Kadj_coeff = self.SimDet.get("epiid_Kadj_coeff", [10.0, 10.0])

        # Original LV normal component used a spatially varying multiplier self.Mesh.poissonF
        # read from the HDF5 dataset "varyingspring" when available; otherwise it is zero in
        # the original mesh helper.  To avoid silently disabling the normal spring if the dataset
        # is absent, allow a fallback value through spring_poisson_fallback.
        poissonF = getattr(self.Mesh, "poissonF", Constant(1.0))
        if self.SimDet.get("spring_poisson_fallback", False):
            poissonF = Constant(float(self.SimDet.get("spring_poisson_fallback", 1.0)))

        u_rate = u_me - u_me_n
        Pn = outer(N_me, N_me)
        Pt = I - Pn

        F3_epi = (
            inner(
                Pn * (k_spring[0] * epiid_Kadj_coeff[0] * poissonF * u_me + c_damping[0] * u_rate),
                v_me,
            ) * ds_me(epiid)
            + inner(
                Pt * (k_spring[1] * epiid_Kadj_coeff[1] * u_me + c_damping[1] * u_rate),
                v_me,
            ) * ds_me(epiid)
        )

        if self.SimDet.get("spring_atbase", False):
            a_, b_ = self.GetTopSpring()
            F3_epi -= a_ * inner(b_, v_me) * ds_me(topid)

        return F3_epi

    def Problem(self):
        GuccioneParams = self.SimDet["GiccioneParams"]
        mesh_me = self.mesh_me
        dx_me = self.dx_me
        ds_me = self.ds_me
        W_me = self.W
        w_me = self.w_me
        w_me_n = self.w_me_n
        dw_me = self.dw_me
        N_me = FacetNormal(mesh_me)
        X_me = SpatialCoordinate(mesh_me)

        if self.use_springbc:
            du, dW, dp = TrialFunctions(W_me)
            u_me, Wflux_me, p_me = split(w_me)
            u_me_n, Wflux_me_n, p_me_n = split(w_me_n)
            v_me, wflux_me, q_me = TestFunctions(W_me)
            c_me = None
        else:
            du, dW, dp, dc = TrialFunctions(W_me)
            u_me, Wflux_me, p_me, c_me = split(w_me)
            u_me_n, Wflux_me_n, p_me_n, c_me_n = split(w_me_n)
            v_me, wflux_me, q_me, cq = TestFunctions(W_me)

        bcs_elas = self.set_BCs()
        LVendoid = self.SimDet["LVendoid"]
        LVPid = self.SimDet.get("LVPid", LVendoid)
        RVendoid = self.SimDet.get("RVendoid", 1000)
        RVPid = self.SimDet.get("RVPid", RVendoid)

        params = {
            "mesh": mesh_me,
            "facetboundaries": self.facetboundaries_me,
            "facet_normal": N_me,
            "mixedfunctionspace": W_me,
            "mixedfunction": w_me,
            "displacement_variable": u_me,
            "pressure_variable": p_me,
            "lv_volconst_variable": [],
            "lv_constrained_vol": self.LVCavityvol,
            "rv_volconst_variable": [],
            "rv_constrained_vol": self.RVCavityvol,
            "LVendoid": LVendoid,
            "RVendoid": RVendoid,
            "epiid": self.SimDet["epiid"],
            "topid": self.SimDet["topid"],
            "aortaid": None,
            "LVPid": LVPid,
            "RVPid": RVPid,
            "aortic_vplane": None,
            "mitral_vplane": None,
            "first_rv_valve": None,
            "second_rv_valve": None,
            "septumid": None,
            "aorta_wall": None,
            "pulm_wall": None,
            "apxid": self.SimDet.get("apxid", None),
            "aorta_int_wall": None,
            "aorta_ext_wall": None,
            "aorta_ring": None,
            "LVendo_comp": 3,
            "RVendo_comp": 1000,
            "fiber": self.f0_me,
            "sheet": self.s0_me,
            "sheet-normal": self.n0_me,
            "growth_tensor": None,
            "material model": GuccioneParams["Passive model"],
            "material params": GuccioneParams["Passive params"],
            "incompressible": GuccioneParams["incompressible"],
            "LVendo_area": self.LVendo_area_me,
            "lv_constrained_pres": self.LVCavitypres,
            "rv_constrained_pres": self.RVCavitypres,
        }
        uflforms = Forms(params)
        self.uflforms = uflforms

        self.t_a = Function(self.Quad)
        self.t_a.vector()[:] = 0.0
        activeparams = {
            "mesh": mesh_me,
            "dx": dx_me,
            "deg": GuccioneParams["deg"],
            "facetboundaries": self.facetboundaries_me,
            "facet_normal": N_me,
            "displacement_variable": u_me,
            "pressure_variable": p_me,
            "fiber": self.f0_me,
            "sheet": self.s0_me,
            "sheet-normal": self.n0_me,
            "t_a": self.t_a,
            "Threshold_Potential": 0.9,
            "growth_tensor": None,
            "material model": GuccioneParams.get("Active model", {"Name": "Time-varying"}),
            "HomogenousActivation": GuccioneParams.get("HomogenousActivation", True),
        }
        if "Active params" in GuccioneParams:
            activeparams["material params"] = GuccioneParams["Active params"]
        self.activeforms = activeForms(activeparams)

        Fmat = uflforms.Fmat()
        Sactive = self.activeforms.PK2StressTensor()
        state_obj = self.parameters["state_obj"]
        pp = self._poro_params()
        permeability = pp["permeability"]
        beta_a = pp["beta_a"]
        beta_v = pp["beta_v"]

        # ------------------------------------------------------------------
        # Three-field finite-strain poroelasticity in the reference domain.
        #
        # Berger et al. formulate the problem on the current configuration
        # using spatial fluid flux z.  Here we use the exact pull-back
        # W = J F^{-1} z (nominal flux), which is compatible with the existing
        # total-Lagrangian cardiac mechanics code:
        #
        #   k0^{-1} W + Grad(p) = 0
        #   Jdot + Div(W) = J s
        #   P_tot = P_eff - J p F^{-T}
        # ------------------------------------------------------------------
        Fmat_poro = uflforms.Fmat()
        J_poro = det(Fmat_poro)
        Fmat_poro_n = Identity(u_me.ufl_shape[0]) + grad(u_me_n)
        J_poro_n = det(Fmat_poro_n)
        dt_ufl = state_obj.dt  # mutable Expression; follows 1/2-ms driver changes

        # Source/sink is interpreted per unit CURRENT volume, hence J*s in the
        # reference-domain mass equation.
        g_comp = self._coronary_arterial_conductance_factor()
        s_cor = beta_a * g_comp * (self.p_a - p_me) - beta_v * (p_me - self.p_v)

        # Darcy first-order equation. No element-wise gradient of DG0 pressure
        # appears. The boundary term is retained because the CG1 flux test space
        # is not restricted strongly to zero normal flux.
        F_darcy = (
            inner((1.0 / permeability) * Wflux_me, wflux_me) * dx_me
            - p_me * div(wflux_me) * dx_me
        )

        flux_ids = self.SimDet.get(
            "flux_boundary_ids",
            [self.SimDet["LVendoid"], self.SimDet["epiid"], self.SimDet["topid"]],
        )
        # Flatten marker lists and remove duplicates.
        _ids = []
        for _mid in flux_ids:
            if isinstance(_mid, list):
                _ids.extend(_mid)
            else:
                _ids.append(_mid)
        _ids = list(dict.fromkeys(int(_mid) for _mid in _ids))
        flux_ds = self._surface_measure(_ids)

        # Consistent boundary term from integration by parts of Grad(p).
        F_darcy += p_me * dot(wflux_me, N_me) * flux_ds

        # Legacy FEniCS adaptation of Berger's normal-flux constraint:
        # impose W.N = 0 weakly without suppressing tangential flux.
        # Berger 2017 uses a boundary Lagrange multiplier; this project uses a
        # normal-only penalty because no mixed-dimensional trace space exists in
        # the original solver infrastructure.  gamma_flux is dimensionless.
        gamma_flux = Constant(float(self.SimDet.get("normal_flux_penalty", 1.0e3)))
        h_cell = CellDiameter(mesh_me)
        F_darcy += (
            gamma_flux * h_cell / permeability
            * dot(Wflux_me, N_me) * dot(wflux_me, N_me) * flux_ds
        )

        # Berger local pressure-jump stabilization.  On legacy FEniCS dS each
        # interior facet is visited once; the scalar difference form below is
        # orientation-independent after multiplying the trial/test jumps.
        dS_me = Measure("dS", domain=mesh_me)
        hF = avg(CellDiameter(mesh_me))
        gamma0 = float(self.SimDet.get("pressure_jump_gamma0", 1.0e-2))
        pref = float(self.SimDet.get("pressure_ref", 13332.0))
        gamma_p = Constant(gamma0 / pref)
        dp_jump = (p_me("+") - p_me("-")) - (p_me_n("+") - p_me_n("-"))
        q_jump = q_me("+") - q_me("-")
        F_stab = gamma_p / dt_ufl * hF * dp_jump * q_jump * dS_me

        F_mass = (
            q_me * (J_poro - J_poro_n) / dt_ufl * dx_me
            + q_me * div(Wflux_me) * dx_me
            + F_stab
            - q_me * J_poro * s_cor * dx_me
        )

        # Total-Lagrangian momentum balance.  A Cauchy pore pressure -p I pulls
        # back to the first Piola term -J p F^{-T}.
        P_pore = J_poro * p_me * inv(Fmat_poro).T
        F_mom = (
            self.LVCavitypres * inner(v_me, N_me) * self._surface_measure(LVPid)
            + inner(grad(v_me), uflforms.poro_PK_1()) * dx_me
            - inner(grad(v_me), P_pore) * dx_me
        )

        # Active stress. Keep active_region hook for future coupling to spatial models.
        if "active_region" in self.SimDet and self.SimDet["active_region"]:
            F_active = Constant(0.0) * q_me * dx_me
            for regionid in self.SimDet["active_region"]:
                F_active += inner(Fmat * Sactive, grad(v_me)) * dx_me(int(regionid))
        else:
            F_active = inner(Fmat * Sactive, grad(v_me)) * dx_me

        if self.use_springbc:
            F_support = self._springbc_residual(u_me, u_me_n, v_me, N_me, ds_me)
        else:
            # Rigid-body constraints retained from the original LV pressure-control branch.
            Wrigid = (
                inner(as_vector([c_me[0], c_me[1], 0.0]), u_me)
                + inner(as_vector([0.0, 0.0, c_me[2]]), cross(X_me, u_me))
                + inner(as_vector([c_me[3], 0.0, 0.0]), cross(X_me, u_me))
                + inner(as_vector([0.0, c_me[4], 0.0]), cross(X_me, u_me))
            )
            F_support = derivative(Wrigid, w_me, self.wtest_me) * dx_me

        Ftotal = F_darcy + F_mass + F_mom + F_active + F_support
        Jac = derivative(Ftotal, w_me, dw_me)
        return Ftotal, Jac, bcs_elas

    # ------------------------------------------------------------------
    # Solver and state management
    # ------------------------------------------------------------------
    def Solver(self):
        solverparams = {
            "Jacobian": self.Jac,
            "F": self.Ftotal,
            "w": self.w_me,
            "boundary_conditions": self.bcs,
            "Type": self.SimDet.get("Type", 0),
            "mesh": self.mesh_me,
            "mode": 1,
        }
        if "abs_tol" in self.SimDet:
            solverparams["abs_tol"] = self.SimDet["abs_tol"]
        if "rel_tol" in self.SimDet:
            solverparams["rel_tol"] = self.SimDet["rel_tol"]
        return NSolver(solverparams)

    def UpdateVar(self):
        self.w_me_n.assign(self.w_me)

    def Reset(self):
        self.w_me.assign(self.w_me_n)

    # ------------------------------------------------------------------
    # Accessors used by the closed-loop driver
    # ------------------------------------------------------------------
    def GetDisplacement(self):
        u, Wflux, p, c = self._current()
        u.rename("u_", "u_")
        return u

    def GetNominalDarcyFlux(self):
        u, Wflux, p, c = self._current()
        Wflux.rename("W_Darcy", "W_Darcy")
        return Wflux

    def GetP(self):
        u, Wflux, p, c = self._current()
        p.rename("p_", "p_")
        return p

    def GetPorePressure(self):
        return self.GetP()

    def GetPorePressure2(self):
        p = self.GetP()
        J = self.uflforms.J()
        vol = assemble(J * self.dx_me)
        if abs(vol) < 1e-30:
            return 0.0
        return assemble(p * J * self.dx_me) / vol

    def GetFmat(self):
        return self.uflforms.Fmat()

    def GetFiberstrain(self, F_ref):
        return self.uflforms.fiberstrain(F_ref=F_ref)

    def GetFiberstrainUL(self):
        F_Identity = Identity(self.GetDisplacement().ufl_domain().geometric_dimension())
        return self.uflforms.fiberstrain(F_ref=F_Identity)

    def GetIMP(self):
        return self.uflforms.IMP()

    def GetIMP2(self):
        return self.uflforms.IMP2()

    def Getfstress(self):
        return self.uflforms.fiberstress() + self.activeforms.fiberstress()

    def GetLVP(self):
        return self.LVCavitypres.pres

    def GetLVV(self):
        return self.LV_closedsurf()

    def LV_closedsurf(self):
        if self.use_springbc:
            return self.uflforms.LVcavityvol_mvb()
        return self.uflforms.LVcavityvol()

    def GetTopSpring(self):
        return self.uflforms.topspringbc()

    def GetRVP(self):
        return 0.0

    def GetRVV(self):
        return 0.0

    def GetWallVolume(self):
        return assemble(self.uflforms.J() * self.dx_me)

    def GetDarcy(self):
        """Backward-compatible accessor: return nominal Darcy flux W."""
        return self.GetNominalDarcyFlux()

    def GetSpatialDarcyFlux(self):
        """Return spatial relative fluid flux z = J^{-1} F W."""
        u, Wflux, p, c = self._current()
        F = Identity(u.ufl_shape[0]) + grad(u)
        J = det(F)
        return (1.0 / J) * F * Wflux

    def GetFlowRate(self):
        u, Wflux, p, c = self._current()
        pp = self._poro_params()
        J = det(Identity(u.ufl_shape[0]) + grad(u))
        g_comp = self._coronary_arterial_conductance_factor()
        return assemble(J * pp["beta_a"] * g_comp * (self.p_a - p) * self.dx_me)

    def GetCoronaryArterialConductanceFactor(self):
        """Return the current scalar g_comp value for runtime diagnostics."""
        g_comp = self._coronary_arterial_conductance_factor()
        volume = assemble(Constant(1.0) * self.dx_me)
        return float(assemble(g_comp * self.dx_me) / max(volume, 1.0e-30))

    def GetFlowRate_n(self):
        u_n, Wflux_n, p_n, c_n = self._previous()
        pp = self._poro_params()
        J_n = det(Identity(u_n.ufl_shape[0]) + grad(u_n))
        return assemble(J_n * pp["beta_v"] * (p_n - self.p_v) * self.dx_me)

    def GetFlowRateEndo(self):
        J = self.uflforms.J()
        area = assemble(Constant(1.0) * self._surface_measure(self.SimDet["LVendoid"]))
        if abs(area) < 1e-30:
            return 0.0
        return assemble((J - 1.0) * self._surface_measure(self.SimDet["LVendoid"])) / area

    def GetMass(self):
        m = project(self.uflforms.J() - 1.0, self.QDG)
        m.rename("m_", "m_")
        return m

    def GetJacobian(self):
        """Return the current deformation Jacobian J without projection."""
        return self.uflforms.J()

    def GetMassGrad(self):
        mg = project(grad(self.uflforms.J() - 1.0), self.W.sub(0).collapse())
        mg.rename("grad_m_", "grad_m_")
        return mg

    def GetNormaldirtctionDarcy(self):
        V0 = FunctionSpace(self.mesh_me, "CG", 1)
        flux = project(div(self.GetDarcy()), V0)
        flux.rename("NormalDarcyFlux_", "NormalDarcyFlux_")
        return flux

    def GetTransmuralCoordinate(self):
        """Return the reference transmural coordinate and its unit normal.

        The harmonic coordinate is zero on the LV endocardium and one on the
        epicardium.  Consequently its unit gradient points from endocardium to
        epicardium.  Both fields are time independent and are cached so this
        diagnostic cannot add repeated Laplace solves to time stepping.
        """
        if hasattr(self, "_transmural_coordinate"):
            return self._transmural_coordinate, self._transmural_normal

        mesh_me = self.mesh_me
        V0 = FunctionSpace(mesh_me, "CG", 1)
        T = Function(V0, name="transmural_coordinate")
        trial = TrialFunction(V0)
        test = TestFunction(V0)
        a = inner(grad(trial), grad(test)) * self.dx_me
        L = Constant(0.0) * test * self.dx_me
        bc_endo = DirichletBC(
            V0, Constant(0.0), self.facetboundaries_me,
            self.SimDet["LVendoid"],
        )
        bc_epi = DirichletBC(
            V0, Constant(1.0), self.facetboundaries_me,
            self.SimDet["epiid"],
        )
        solve(
            a == L, T, [bc_endo, bc_epi],
            solver_parameters={"linear_solver": "mumps"},
        )

        Wv = VectorFunctionSpace(mesh_me, "CG", 1)
        gT = project(
            grad(T), Wv, solver_type="mumps",
            form_compiler_parameters={"representation": "uflacs"},
        )
        eps = Constant(1.0e-24)
        n_tm = project(
            gT / sqrt(inner(gT, gT) + eps), Wv, solver_type="mumps",
            form_compiler_parameters={"representation": "uflacs"},
        )
        n_tm.rename("transmural_normal", "transmural_normal")

        self._transmural_coordinate = T
        self._transmural_normal = n_tm
        return self._transmural_coordinate, self._transmural_normal

    def GetNormaldirtction(self):
        """Backward-compatible alias for the legacy misspelled accessor."""
        return self.GetTransmuralCoordinate()

    def _get_transmural_layer_indicators(self):
        """Return cached DG0 indicators for the three reference wall thirds.

        The harmonic coordinate is projected once to DG0 so each tetrahedron is
        assigned to exactly one material layer. No indicator enters the
        nonlinear residual or changes a solution field.
        """
        if hasattr(self, "_transmural_layer_indicators"):
            return self._transmural_layer_indicators

        Ttm, _ = self.GetTransmuralCoordinate()
        T_cell = project(
            Ttm, self.QDG, solver_type="mumps",
            form_compiler_parameters={"representation": "uflacs"},
        )
        values = T_cell.vector().get_local()
        masks = {
            "endo": values < (1.0 / 3.0),
            "mid": ((values >= (1.0 / 3.0)) & (values < (2.0 / 3.0))),
            "epi": values >= (2.0 / 3.0),
        }
        indicators = {}
        for name, mask in masks.items():
            indicator = Function(self.QDG, name="chi_{}".format(name))
            local = np.zeros_like(values)
            local[mask] = 1.0
            indicator.vector().set_local(local)
            indicator.vector().apply("insert")
            indicators[name] = indicator

        self._transmural_layer_indicators = indicators
        return indicators

    def GetRegionalTransmuralDiagnostics(self):
        """Return read-only mass-balance terms for three reference wall thirds.

        ``fluid_content`` is integral_(layer) (J-1) dV0 and therefore has the
        units of density-normalized fluid mass (mL for a cm-based mesh).
        Source, sink, storage, divergence, stabilization, and residual are
        rates. The stabilization sign matches the discrete mass residual:

          storage + flux_divergence + stabilization
                  - arterial_source + venous_sink = residual.
        """
        u, Wflux, p, c = self._current()
        u_n, Wflux_n, p_n, c_n = self._previous()
        dim = u.ufl_shape[0]
        J = det(Identity(dim) + grad(u))
        J_n = det(Identity(dim) + grad(u_n))
        dt_value = float(self.parameters["state_obj"].dt.dt)
        pp = self._poro_params()
        g_comp = self._coronary_arterial_conductance_factor()

        storage = (J - J_n) / dt_value
        flux_divergence = div(Wflux)
        arterial_source = J * pp["beta_a"] * g_comp * (self.p_a - p)
        venous_sink = J * pp["beta_v"] * (p - self.p_v)

        hF = avg(CellDiameter(self.mesh_me))
        dS_me = Measure("dS", domain=self.mesh_me)
        gamma0 = float(self.SimDet.get("pressure_jump_gamma0", 1.0e-2))
        pref = float(self.SimDet.get("pressure_ref", 13332.0))
        gamma_p = Constant(gamma0 / pref)
        dp_jump = (p("+") - p("-")) - (p_n("+") - p_n("-"))

        diagnostics = {}
        for name, chi in self._get_transmural_layer_indicators().items():
            chi_jump = chi("+") - chi("-")
            stabilization = assemble(
                gamma_p / dt_value * hF * dp_jump * chi_jump * dS_me
            )
            terms = {
                "reference_volume": float(assemble(chi * self.dx_me)),
                "fluid_content": float(assemble((J - 1.0) * chi * self.dx_me)),
                "storage": float(assemble(storage * chi * self.dx_me)),
                "flux_divergence": float(assemble(
                    flux_divergence * chi * self.dx_me
                )),
                "stabilization": float(stabilization),
                "arterial_source": float(assemble(
                    arterial_source * chi * self.dx_me
                )),
                "venous_sink": float(assemble(venous_sink * chi * self.dx_me)),
            }
            terms["residual"] = (
                terms["storage"] + terms["flux_divergence"]
                + terms["stabilization"] - terms["arterial_source"]
                + terms["venous_sink"]
            )
            diagnostics[name] = terms
        return diagnostics

    def GetPressureJumpNorm(self):
        """Mesh-scaled L2 norm of the DG0 pressure jump on interior facets."""
        p = self.GetP()
        hF = avg(CellDiameter(self.mesh_me))
        dS_me = Measure("dS", domain=self.mesh_me)
        jp = p("+") - p("-")
        val = assemble(hF * jp * jp * dS_me)
        return float(sqrt(max(val, 0.0)))

    def GetBoundaryNormalFluxRMS(self):
        """RMS nominal normal-flux leakage on the impermeable boundary."""
        Wflux = self.GetNominalDarcyFlux()
        N = FacetNormal(self.mesh_me)
        flux_ids = self.SimDet.get(
            "flux_boundary_ids",
            [self.SimDet["LVendoid"], self.SimDet["epiid"], self.SimDet["topid"]],
        )
        ids = []
        for mid in flux_ids:
            ids.extend(mid if isinstance(mid, list) else [mid])
        ids = list(dict.fromkeys(int(mid) for mid in ids))
        ds_flux = self._surface_measure(ids)
        area = assemble(Constant(1.0) * ds_flux)
        if abs(area) < 1e-30:
            return 0.0
        val = assemble(dot(Wflux, N) ** 2 * ds_flux) / area
        return float(np.sqrt(max(val, 0.0)))

    def GetMassBalanceDiagnostics(self):
        """Return signed/absolute global backward-Euler mass-balance terms.

        The relative residual is normalized by the sum of the L1 magnitudes of
        storage, flux divergence, arterial source, and venous sink.  This is
        well-defined even when signed global terms nearly cancel internally.
        """
        u, Wflux, p, c = self._current()
        u_n, Wflux_n, p_n, c_n = self._previous()
        dim = u.ufl_shape[0]
        J = det(Identity(dim) + grad(u))
        J_n = det(Identity(dim) + grad(u_n))
        pp = self._poro_params()
        dt_value = float(self.parameters["state_obj"].dt.dt)
        storage = (J - J_n) / dt_value
        flux_divergence = div(Wflux)
        g_comp = self._coronary_arterial_conductance_factor()
        arterial_source = J * pp["beta_a"] * g_comp * (self.p_a - p)
        venous_sink = J * pp["beta_v"] * (p - self.p_v)
        residual = storage + flux_divergence - arterial_source + venous_sink

        signed = {
            "storage": float(assemble(storage * self.dx_me)),
            "flux_divergence": float(assemble(flux_divergence * self.dx_me)),
            "boundary_flux": float(assemble(
                dot(Wflux, FacetNormal(self.mesh_me))
                * Measure("ds", domain=self.mesh_me)
            )),
            "arterial_source": float(assemble(arterial_source * self.dx_me)),
            "venous_sink": float(assemble(venous_sink * self.dx_me)),
            "residual": float(assemble(residual * self.dx_me)),
        }
        scale = float(assemble(
            (abs(storage) + abs(flux_divergence) + abs(arterial_source)
             + abs(venous_sink)) * self.dx_me
        ))
        signed["absolute_scale"] = scale
        signed["relative_residual"] = abs(signed["residual"]) / max(scale, 1.0e-30)
        signed["divergence_theorem_error"] = (
            signed["flux_divergence"] - signed["boundary_flux"]
        )
        return signed

    def GetGlobalMassBalanceResidual(self):
        """Backward-compatible signed global mass-balance residual."""
        return self.GetMassBalanceDiagnostics()["residual"]

    def GetPorePressureStatistics(self):
        """MPI-safe DG0 pressure min/max and volume-weighted mean."""
        p = self.GetP()
        local = p.vector().get_local()
        comm = self.mesh_me.mpi_comm()
        local_min = float(np.min(local)) if local.size else np.inf
        local_max = float(np.max(local)) if local.size else -np.inf
        pmin = MPI.min(comm, local_min)
        pmax = MPI.max(comm, local_max)
        volume = assemble(Constant(1.0) * self.dx_me)
        pmean = assemble(p * self.dx_me) / max(volume, 1.0e-30)
        return {"min": float(pmin), "max": float(pmax), "mean": float(pmean)}

    def GetFluxStatistics(self):
        """Volume-normalized RMS and global maximum for nominal/spatial flux."""
        W = self.GetNominalDarcyFlux()
        z = self.GetSpatialDarcyFlux()
        volume = assemble(Constant(1.0) * self.dx_me)
        comm = self.mesh_me.mpi_comm()

        def _stats(field):
            rms = sqrt(max(assemble(inner(field, field) * self.dx_me) / max(volume, 1.0e-30), 0.0))
            magnitude = project(
                sqrt(inner(field, field)), FunctionSpace(self.mesh_me, "DG", 0),
                solver_type="mumps",
                form_compiler_parameters={"representation": "uflacs"},
            )
            values = magnitude.vector().get_local()
            local_max = float(np.max(values)) if values.size else 0.0
            return float(rms), float(MPI.max(comm, local_max))

        nominal_rms, nominal_max = _stats(W)
        spatial_rms, spatial_max = _stats(z)
        return {
            "nominal_rms": nominal_rms, "nominal_max": nominal_max,
            "spatial_rms": spatial_rms, "spatial_max": spatial_max,
        }

    def GetMechanicsDOFs(self):
        """Total number of algebraic unknowns in the three-field mechanics solve."""
        return int(self.W.dim())

    def GetSActive(self):
        Sactive = self.activeforms.PK2StressTensor()
        i, j = indices(2)
        Sactive_ = project(self.f0_me[i] * Sactive[i, j] * self.f0_me[j], self.QDG)
        Sactive_.rename("Sact", "Sact")
        return Sactive_

    # ------------------------------------------------------------------
    # Kept as explicit unsupported hooks for old postprocessing paths.
    # ------------------------------------------------------------------
    def GetFiberNaturalStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()
        return self.activeforms.CalculateFiberNaturalStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetFiberBiotStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()
        return self.activeforms.CalculateFiberBiotStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetFiberGreenStrain(self, F_ED, basis_dir, AHA_segments):
        F_n = self.GetFmat()
        return self.activeforms.CalculateFiberGreenStrain(
            F_=F_n, F_ref=F_ED, e_fiber=basis_dir, VolSeg=AHA_segments
        )

    def GetDeformedBasis(self, params):
        raise NotImplementedError(
            "GetDeformedBasis/fiber regeneration was removed from the minimal LV poroelastic model. "
            "Use the original MEmodel3.py if you need unloading/fiber regeneration."
        )
