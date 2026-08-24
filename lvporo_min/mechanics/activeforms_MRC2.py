from dolfin import *
import sys
import math
import numpy as np

from ufl import indices


class activeForms(object):
    def __init__(self, params):
        self.parameters = self.default_parameters()
        # print self.parameters
        self.parameters.update(params)
        # print self.parameters

        Matmodel = self.parameters["material model"]["Name"]
        assert (
            Matmodel == "Guccione" or Matmodel == "Time-varying"
        ), "Material model not implemented"

        # self.parameters.update({"F": self.Fe()})
        if Matmodel == "Guccione":
            from .GuccioneAct import GuccioneAct as Active

        if Matmodel == "Time-varying":
            from .BurkhoffTimevarying3 import BurkhoffTimevarying as Active

        deg = self.parameters["deg"]
        bivMesh = self.parameters["mesh"]
        mesh = bivMesh

        Quadelem = FiniteElement(
            "Quadrature", mesh.ufl_cell(), degree=deg, quad_scheme="default"
        )
        Quadelem._quad_scheme = "default"
        self.Quadelem = Quadelem
        self.Quad = FunctionSpace(mesh, Quadelem)

        self.t_init = self.get_t_init()
        self.isActive = self.get_isActive()

        self.parameters.update(
            {"t_init": self.t_init, "isActive": self.isActive, "Fmat": self.Fmat()}
        )

        self.activeforms = Active(self.parameters)
        self.matparams = self.activeforms.Getmatparam()

    def default_parameters(self):
        return {"material model": {"Name": "Time-varying"}}

    def get_isActive(self):
        deg = self.parameters["deg"]
        mesh = self.parameters["mesh"]

        Quad = self.Quad
        isActive = Function(Quad)

        isActive_array = isActive.vector().get_local()
        isActive_array = np.zeros(len(isActive_array))

        isActive.vector()[:] = isActive_array

        return isActive

    def Get_t_a(self):
        return self.parameters["t_a"].ta

    def update_activationTime(self, potential_n, comm):
        """
        if V[i] >= V_thres[i] and isActivation[i] == 0
            then t0[i] = t_a
        """

        mesh = self.parameters["mesh"]

        V_thres = self.parameters["Threshold_Potential"]
        current_ta = self.parameters["t_a"]
        current_ta_array = interpolate(current_ta, self.Quad).vector().get_local()

        t_init_array = self.t_init.vector().get_local()
        isActive_array = self.isActive.vector().get_local()

        phi_n_interp = interpolate(potential_n, self.Quad)

        V_n_array = phi_n_interp.vector().get_local()

        tol_isActive = 1e-1
        comm.Barrier()
        for idx, (vn, isactive, tinit, cur_t) in enumerate(
            zip(V_n_array, isActive_array, t_init_array, current_ta_array)
        ):
            if (abs(isactive) <= tol_isActive) and vn >= V_thres:
                isActive_array[idx] = 1.0
                t_init_array[idx] = current_ta_array[idx]

        comm.Barrier()

        # If homogeneous ---------------------------------------------------
        isHomogenousActivation = self.parameters["HomogenousActivation"]
        # print "isHomogeneous = ", isHomogenousActivation
        if isHomogenousActivation:  # activation at 10 ms
            t_init_array = 0.0 * np.ones(len(self.t_init.vector().get_local()))
        # -------------------------------------------------------------------

        self.t_init.vector()[:] = t_init_array
        self.isActive.vector()[:] = isActive_array

    def get_t_init(self):
        Quad = self.Quad

        t_init = Function(Quad)

        t_init_array = t_init.vector().get_local()
        t_init_array = 9999.0 * np.ones(len(t_init_array))

        t_init.vector()[:] = t_init_array

        return t_init

    def restart_t_init(self):
        self.t_init = self.get_t_init()
        self.isActive = self.get_isActive()

    def Fmat(self):
        u = self.parameters["displacement_variable"]
        d = u.ufl_domain().geometric_dimension()
        I = Identity(d)
        F = I + grad(u)
        return F

    def Fe(self):
        Fg = self.parameters["growth_tensor"]
        F = self.Fmat()

        if Fg is None:
            Fe = F
        else:
            Fe = as_tensor(F[i, j] * inv(Fg)[j, k], (i, k))
        return Fe

    def GetPact(self):
        return self.activeforms.PK1Stress()

    def PK2StressTensor(self):
        F = self.Fmat()
        f0 = self.parameters["fiber"]
        i, j = indices(2)
        Mij = f0[i] * f0[j]

        # Pact = self.PK1Stress()
        Sact = self.activeforms.PK2Stress()

        Pact_tensor = Sact * as_tensor(Mij, (i, j))

        return Pact_tensor

    def fiberstress(self):
        PK2 = self.PK2StressTensor()
        F = self.Fe()
        f0 = self.parameters["fiber"]
        J = det(F)

        # Tca = (1.0/J)*PK1*F.T
        # Sca = inv(F)*PK1
        i, j = indices(2)
        return f0[i] * PK2[i, j] * f0[j]

    def CalculateFiberNaturalStrain(self, F_, F_ref, e_fiber, VolSeg):
        I = Identity(F_.ufl_domain().geometric_dimension())

        # Pull back to ED
        F = F_ * inv(F_ref)
        # Right Cauchy Green
        C = F.T * F

        C_fiber = inner(C * e_fiber, e_fiber)
        E_fiber = 0.5 * (1 - 1 / C_fiber)

        mesh = self.parameters["mesh"]
        dx = self.parameters["dx"]

        E_fiber_BiV = [
            assemble(
                E_fiber * dx(ii), form_compiler_parameters={"representation": "uflacs"}
            )
            for ii in VolSeg
        ]

        return E_fiber_BiV, E_fiber
