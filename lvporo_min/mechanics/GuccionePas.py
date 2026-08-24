from dolfin import *
from ufl import indices
import dolfin as dolfin
import collections.abc


class GuccionePas(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        # self.parameters.update(params)
        self.update(self.parameters, params)

    def update(self, d, u):
        for k, v in u.items():
            if isinstance(v, collections.abc.Mapping):
                d[k] = self.update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    def default_parameters(self):
        return {
            "material params": {
                "bff": Constant(29.9),
                "bfx": Constant(13.3),
                "bxx": Constant(26.6),
                "Cparam": Constant(100),
                "mu_iso": Constant(5e4),
                "b_iso": Constant(10),
            },
            "porous params": {
                "Ks": Constant(5.0e4),
                "phi0": Constant(0.06),
                "c1": Constant(1.33),
                "c2": Constant(550.0),
                "c3": Constant(45.0),
            },
        }

    def Getmatparam(self):
        return self.parameters["material params"]

    def Emat(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = self.parameters["F"]
        i, j, k = indices(3)
        Emat = 0.5 * (as_tensor(F[k, i] * F[k, j] - I[i, j], (i, j)))

        return Emat

    # def PassiveRubSEF(self):
    #     u = self.parameters["displacement_variable"]
    #     d = u.ufl_domain().geometric_dimension()
    #     I = Identity(d)
    #     F = I + grad(u)
    #     F = dolfin.variable(F)
    #     J = det(F)
    #     Ic = tr(F.T * F)
    #     C = self.parameters["material params"]["Cparam"]
    #     #mu = Constant(5e4)
    #     mu = self.parameters["material params"]["mu_iso"]
    #     b_iso = self.parameters["material params"]["b_iso"]
    #     # Wp = (mu / 2) * (Ic - 3)  # - mu*ln(J)
    #     # Wp = (mu / 2) * (exp(b_iso * (Ic - 3) * (Ic - 3)) - 1) # LCL # - p * (J - 1)
    #     # Wp = (mu / 2) * (exp(b_iso * (Ic - 3)) - 1) # LCL # - p * (J - 1)
    #     Wp = C * (exp(b_iso * (Ic - 3)) - 1)

    #     return Wp

    def PassiveRubSEF(self):
        Ea = self.Emat()
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]

        b_iso = self.parameters["material params"]["b_iso"]
        isincomp = self.parameters["incompressible"]

        if isincomp:
            p = self.parameters["pressure_variable"]

        C = self.parameters["material params"]["Cparam"]

        QQ = b_iso * inner(Ea, Ea)
        Wp = C / 2.0 * (exp(QQ) - 1.0)

        return Wp

    def PassiveAortaSEF(self):
        Ea = self.Emat()
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]

        b_iso = self.parameters["material params"]["b_iso"]
        isincomp = self.parameters["incompressible"]

        if isincomp:
            p = self.parameters["pressure_variable"]

        C = self.parameters["material params"]["Cparam"]
        if ("aorta_comp_red") in self.parameters["material params"]:
            compliance_reduction = self.parameters["material params"]["aorta_comp_red"]
        else:
            compliance_reduction = 1.0  # default

        QQ = compliance_reduction * b_iso * inner(Ea, Ea)
        Wp = C / 2.0 * (exp(QQ) - 1.0)

        return Wp

    def PassiveMatSEF(self):
        Ea = self.Emat()
        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]
        bff = self.parameters["material params"]["bff"]
        bfx = self.parameters["material params"]["bfx"]
        bxx = self.parameters["material params"]["bxx"]
        isincomp = self.parameters["incompressible"]

        if isincomp:
            p = self.parameters["pressure_variable"]

        C = self.parameters["material params"]["Cparam"]

        Eff = inner(f0, Ea * f0)
        Ess = inner(s0, Ea * s0)
        Enn = inner(n0, Ea * n0)
        Efs = inner(f0, Ea * s0)
        Efn = inner(f0, Ea * n0)
        Ens = inner(n0, Ea * s0)
        Esf = inner(s0, Ea * f0)
        Enf = inner(n0, Ea * f0)
        Esn = inner(s0, Ea * n0)

        QQ = (
            bff * Eff**2.0
            + bxx * (Ess**2.0 + Enn**2.0 + Ens**2.0 + Esn**2.0)
            + bfx * (Efs**2.0 + Esf**2.0 + Efn**2.0 + Enf**2.0)
        )

        Wp = C / 2.0 * (exp(QQ) - 1.0)

        return Wp

    def PK1(self):
        u = self.parameters["displacement_variable"]

        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]
        bff = self.parameters["material params"]["bff"]
        bfx = self.parameters["material params"]["bfx"]
        bxx = self.parameters["material params"]["bxx"]
        C = self.parameters["material params"]["Cparam"]
        # p = self.parameters["pressure_variable"]

        #### For some reason to use dolfin.diff, you need to declare everything starting from u #############################
        d = u.geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        F = dolfin.variable(F)
        J = det(F)

        i, j, k = indices(3)
        Ea = 0.5 * (as_tensor(F[k, i] * F[k, j] - I[i, j], (i, j)))

        Eff = inner(f0, Ea * f0)
        Ess = inner(s0, Ea * s0)
        Enn = inner(n0, Ea * n0)
        Efs = inner(f0, Ea * s0)
        Efn = inner(f0, Ea * n0)
        Ens = inner(n0, Ea * s0)
        Esf = inner(s0, Ea * f0)
        Enf = inner(n0, Ea * f0)
        Esn = inner(s0, Ea * n0)

        QQ = (
            bff * Eff**2.0
            + bxx * (Ess**2.0 + Enn**2.0 + Ens**2.0 + Esn**2.0)
            + bfx * (Efs**2.0 + Esf**2.0 + Efn**2.0 + Enf**2.0)
        )
        Wp = C / 2.0 * (exp(QQ) - 1.0)  # - p*(J - 1.0)

        PK1 = dolfin.diff(Wp, F)
        return PK1
    
    def fibermat(self):
        u = self.parameters["displacement_variable"]

        f0 = self.parameters["fiber"]
        s0 = self.parameters["sheet"]
        n0 = self.parameters["sheet-normal"]
        # p = self.parameters["pressure_variable"]

        #### For some reason to use dolfin.diff, you need to declare everything starting from u #############################
        d = u.geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        F = dolfin.variable(F)
        J = det(F)
        Mat = as_tensor([[f0[i] * f0[j] for j in range(d)] for i in range(d)])
        return Mat,I


    def poro_volumetricstress(self):
        phi0 = self.parameters["porous params"]["phi0"]
        Ks = self.parameters["porous params"]["Ks"]
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = variable(I + grad(u))
        J = det(F)
        #psi_skel = (Ks/2) * (J - 1)*ln(J)
        psi_skel = (Ks/2) * (J - 1)**2
        return diff(psi_skel, F)

    def poro_porositystress(self):
        Ks = self.parameters["porous params"]["Ks"]
        phi = self.parameters["pressure_variable"]
        phi0 = self.parameters["porous params"]["phi0"]
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = variable(I + grad(u))
        J = det(F)
        phi_s = variable(J - phi)
        # phi_s0 = 1.0 - phi0
        psi_s = Ks * (
            phi_s - (1 - phi0) - ln(phi_s / (1 - phi0))
        )  # with phi_s ~= phi_s0 constrain
        # psi_s = self.Ks*(phi_s-phi_s0)
        P1 = diff(psi_s, phi_s) * J * inv(F.T)
        return P1

    def poro_pressure(self):
        Ks = self.parameters["porous params"]["Ks"]
        phi0 = self.parameters["porous params"]["phi0"]
        c1 = self.parameters["porous params"]["c1"]
        c2 = self.parameters["porous params"]["c2"]
        c3 = self.parameters["porous params"]["c3"]
        u = self.parameters["displacement_variable"]
        phi = self.parameters["pressure_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = variable(I + grad(u))
        J = det(F)
        psi = c1 * exp(c3 * phi) + c2 * ln(c3 * phi)
        phi_s = variable(J - phi)
        # phi_s0 = 1.0 - phi0
        psi_s = Ks * (
            phi_s - (1 - phi0) - ln(phi_s / (1 - phi))
        )  # with phi_s ~= phi_s0 constrain
        # psi_s = Ks * (phi_s - (1 - phi0))
        p1 = diff(psi, variable(phi))
        p2 = diff(psi_s, phi_s)
        return p1 - p2
