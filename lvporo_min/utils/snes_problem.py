from dolfin import *

class SNESProblem:
    def __init__(self, F, u, bcs):
        V = u.function_space()
        du = TrialFunction(V)
        self.L = F
        self.a = derivative(F, u, du)
        self.bcs = bcs
        self.u = u

    def F(self, snes, x, F):
        x = PETScVector(x)
        F = PETScVector(F)

        # to run in parallel
        x.vec().copy(self.u.vector().vec())
        self.u.vector().apply("")

        assemble(self.L, tensor=F)
        for bc in self.bcs:
            bc.apply(F, x)
            # to run in parallel
            bc.apply(F, self.u.vector())

    def J(self, snes, x, J, P):
        J = PETScMatrix(J)
        # to run in parallel
        x.copy(self.u.vector().vec())
        self.u.vector().apply("")

        assemble(self.a, tensor=J)
        for bc in self.bcs:
            bc.apply(J)
