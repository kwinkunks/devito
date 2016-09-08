import numpy as np
from sympy import Eq, Function, Matrix, solve, symbols
from sympy.abc import p

from devito.dimension import t
from devito.interfaces import DenseData, PointData, TimeData
from devito.iteration import Iteration
from devito.operator import *


class SourceLike(PointData):
    """Defines the behaviour of sources and receivers.
    """
    def __init__(self, *args, **kwargs):
        self.dt = kwargs.get('dt')
        self.h = kwargs.get('h')
        self.ndim = kwargs.get('ndim')
        self.nbpml = kwargs.get('nbpml')
        PointData.__init__(self, *args, **kwargs)
        x1, y1, z1, x2, y2, z2 = symbols('x1, y1, z1, x2, y2, z2')

        if self.ndim == 2:
            A = Matrix([[1, x1, z1, x1*z1],
                        [1, x1, z2, x1*z2],
                        [1, x2, z1, x2*z1],
                        [1, x2, z2, x2*z2]])
            self.increments = (0, 0), (0, 1), (1, 0), (1, 1)
            self.rs = symbols('rx, rz')
            rx, rz = self.rs
            p = Matrix([[1],
                        [rx],
                        [rz],
                        [rx*rz]])
        else:
            A = Matrix([[1, x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1],
                        [1, x1, y2, z1, x1*y2, x1*z1, y2*z1, x1*y2*z1],
                        [1, x2, y1, z1, x2*y1, x2*z1, y2*z1, x2*y1*z1],
                        [1, x1, y1, z2, x1*y1, x1*z2, y1*z2, x1*y1*z2],
                        [1, x2, y2, z1, x2*y2, x2*z1, y2*z1, x2*y2*z1],
                        [1, x1, y2, z2, x1*y2, x1*z2, y2*z2, x1*y2*z2],
                        [1, x2, y1, z2, x2*y1, x2*z2, y1*z2, x2*y1*z2],
                        [1, x2, y2, z2, x2*y2, x2*z2, y2*z2, x2*y2*z2]])
            self.increments = (0, 0, 0), (0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), \
                              (0, 1, 1), (1, 0, 1), (1, 1, 1)
            self.rs = symbols('rx, ry, rz')
            rx, ry, rz = self.rs
            p = Matrix([[1],
                        [rx],
                        [ry],
                        [rz],
                        [rx*ry],
                        [rx*rz],
                        [ry*rz],
                        [rx*ry*rz]])

        # Map to reference cell
        reference_cell = [(x1, 0),
                          (y1, 0),
                          (z1, 0),
                          (x2, self.h),
                          (y2, self.h),
                          (z2, self.h)]
        A = A.subs(reference_cell)
        self.bs = A.inv().T.dot(p)

    @property
    def sym_coordinates(self):
        """Symbol representing the coordinate values in each dimension"""
        return tuple([self.coordinates.indexed[p, i]
                      for i in range(self.ndim)])

    @property
    def sym_coord_indices(self):
        """Symbol for each grid index according to the coordinates"""
        return tuple([Function('INT')(Function('floor')(x / self.h))
                      for x in self.sym_coordinates])

    @property
    def sym_coord_bases(self):
        """Symbol for the base coordinates of the reference grid point"""
        return tuple([Function('FLOAT')(x - idx * self.h)
                      for x, idx in zip(self.sym_coordinates,
                                        self.sym_coord_indices)])

    def point2grid(self, u, m, t):
        """Generates an expression for generic point-to-grid interpolation"""
        dt = self.dt
        subs = dict(zip(self.rs, self.sym_coord_bases))
        index_matrix = [tuple([idx + ii + self.nbpml for ii, idx
                               in zip(inc, self.sym_coord_indices)])
                        for inc in self.increments]
        eqns = [Eq(u.indexed[(t, ) + idx], u.indexed[(t, ) + idx]
                   + self.indexed[t, p] * dt * dt / m.indexed[idx] * b.subs(subs))
                for idx, b in zip(index_matrix, self.bs)]
        return eqns

    def grid2point(self, u, t=t):
        """Generates an expression for generic grid-to-point interpolation"""
        subs = dict(zip(self.rs, self.sym_coord_bases))
        index_matrix = [tuple([idx + ii + self.nbpml for ii, idx
                               in zip(inc, self.sym_coord_indices)])
                        for inc in self.increments]
        return sum([b.subs(subs) * u.indexed[(t, ) + idx]
                    for idx, b in zip(index_matrix, self.bs)])

    def read(self, u):
        """Iteration loop over points performing grid-to-point interpolation."""
        interp_expr = Eq(self.indexed[t, p], self.grid2point(u))
        return [Iteration(interp_expr, index=p, limits=self.shape[1])]

    def read2(self, u, v):
        """Iteration loop over points performing grid-to-point interpolation."""
        interp_expr = Eq(self.indexed[t, p], self.grid2point(u) + self.grid2point(v))
        return [Iteration(interp_expr, index=p, limits=self.shape[1])]

    def add(self, m, u, t=t):
        """Iteration loop over points performing point-to-grid interpolation."""
        return [Iteration(self.point2grid(u, m, t), index=p, limits=self.shape[1])]


class ForwardOperator(Operator):
    def __init__(self, model, src, damp, data, time_order=2, spc_order=6,
                 save=False, **kwargs):
        nt, nrec = data.traces.shape
        nt, nsrc = src.traces.shape
        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order, save=save,
                     dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        u.pad_time = save
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        source = SourceLike(name="src", npoint=nsrc, nt=nt, dt=dt, h=model.get_spacing(),
                            coordinates=src.receiver_coords, ndim=len(damp.shape),
                            dtype=damp.dtype, nbpml=model.nbpml)
        source.data[:] = src.traces[:]
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * u.dx2 + (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 + (1/rho)**2 * rho.dy * u.dy +
                       1/rho * u.dz2 + (1/rho)**2 * rho.dz * u.dz)
            else:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy)
        else:
            Lap = u.laplace
            rho = 1
            # Derive stencil from symbolic equation
        eqn = m / rho * u.dt2 - Lap + damp * u.dt
        stencil = solve(eqn, u.forward)[0]
        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: dt, h: model.get_spacing()}
        super(ForwardOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(u.forward, stencil),
                                              subs=subs,
                                              spc_border=spc_order/2,
                                              time_order=time_order,
                                              forward=True,
                                              dtype=m.dtype,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [source, source.coordinates, rec, rec.coordinates]
        self.output_params += [rec]
        self.propagator.time_loop_stencils_a = source.add(m, u) + rec.read(u)
        self.propagator.add_devito_param(source)
        self.propagator.add_devito_param(source.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class AOperator(Operator):
    def __init__(self, model, u, damp, time_order=2, spc_order=6,
                 **kwargs):
        dt = model.get_critical_dt()
        q = TimeData(name="q", shape=model.get_shape_comp(), time_dim=u.shape[0]-2,
                     time_order=time_order, space_order=spc_order, save=True,
                     dtype=damp.dtype)
        q.pad_time = True
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * u.dx2 + (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 + (1/rho)**2 * rho.dy * u.dy +
                       1/rho * u.dz2 + (1/rho)**2 * rho.dz * u.dz)
            else:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy)
        else:
            Lap = u.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * u.dt2 - Lap + damp * u.dt
        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: dt, h: model.get_spacing()}
        super(AOperator, self).__init__(u.shape[0]-2, m.shape,
                                        stencils=Eq(q.forward, eqn),
                                        subs=subs,
                                        spc_border=spc_order/2,
                                        time_order=time_order,
                                        forward=True,
                                        dtype=m.dtype,
                                        **kwargs)


class AdjointOperator(Operator):
    def __init__(self, model, damp, data, src, recin,
                 time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.traces.shape
        dt = model.get_critical_dt()
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        srca = SourceLike(name="srca", npoint=src.traces.shape[1],
                          nt=nt, dt=dt, h=model.get_spacing(),
                          coordinates=src.receiver_coords,
                          ndim=len(damp.shape), dtype=damp.dtype, nbpml=model.nbpml)
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            Lap = v.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * v.dt2 - Lap - damp * v.dt
        stencil = solve(eqn, v.backward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        super(AdjointOperator, self).__init__(nt, m.shape,
                                              stencils=Eq(v.backward, stencil),
                                              subs=subs,
                                              spc_border=spc_order/2,
                                              time_order=time_order,
                                              forward=False,
                                              dtype=m.dtype,
                                              **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [srca, srca.coordinates, rec, rec.coordinates]
        self.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)
        self.output_params = [srca]
        self.propagator.add_devito_param(srca)
        self.propagator.add_devito_param(srca.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class AadjOperator(Operator):
    def __init__(self, model, v, damp, time_order=2, spc_order=6, **kwargs):
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=v.shape[0]-2,
                     time_order=time_order, space_order=spc_order,
                     save=True, dtype=damp.dtype)
        u.pad_time = True
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            Lap = v.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * v.dt2 - Lap - damp * v.dt
        stencil = Eq(u.backward, eqn)
        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        super(AadjOperator, self).__init__(v.shape[0]-4, m.shape,
                                           stencils=stencil,
                                           subs=subs,
                                           spc_border=spc_order/2,
                                           time_order=time_order,
                                           forward=False,
                                           dtype=m.dtype,
                                           **kwargs)


class GradientOperator(Operator):
    def __init__(self, model, damp, data, recin, u, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.traces.shape
        dt = model.get_critical_dt()
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin
        grad = DenseData(name="grad", shape=m.shape, dtype=m.dtype)
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            Lap = v.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * v.dt2 - Lap - damp * v.dt
        stencil = solve(eqn, v.backward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        # Add Gradient-specific updates. The dt2 is currently hacky
        #  as it has to match the cyclic indices
        gradient_update = Eq(grad, grad - s**-2*(v + v.forward - 2 * v.forward.forward) *
                             u.forward)
        stencils = [gradient_update, Eq(v.backward, stencil)]
        super(GradientOperator, self).__init__(rec.nt - 1, m.shape,
                                               stencils=stencils,
                                               subs=[subs, subs, {}],
                                               spc_border=spc_order/2,
                                               time_order=time_order,
                                               forward=False,
                                               dtype=m.dtype,
                                               input_params=[m, v, damp, u],
                                               **kwargs)
        # Insert receiver term post-hoc
        self.input_params += [grad, rec, rec.coordinates]
        self.output_params = [grad]
        self.propagator.time_loop_stencils_b = rec.add(m, v, t + 1)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class BornOperator(Operator):
    def __init__(self, model, src, damp, data, dmin, time_order=2, spc_order=6, **kwargs):
        nt, nrec = data.traces.shape
        nt, nsrc = src.traces.shape
        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        U = TimeData(name="U", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()

        dm = DenseData(name="dm", shape=model.get_shape_comp(), dtype=damp.dtype)
        dm.data[:] = model.pad(dmin)

        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        source = SourceLike(name="src", npoint=nsrc, nt=nt, dt=dt, h=model.get_spacing(),
                            coordinates=src.receiver_coords, ndim=len(damp.shape),
                            dtype=damp.dtype, nbpml=model.nbpml)
        source.data[:] = src.traces[:]
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * u.dx2 + (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 + (1/rho)**2 * rho.dy * u.dy +
                       1/rho * u.dz2 + (1/rho)**2 * rho.dz * u.dz)
                LapU = (1/rho * U.dx2 + (1/rho)**2 * rho.dx * U.dx +
                        1/rho * U.dy2 + (1/rho)**2 * rho.dy * U.dy +
                        1/rho * U.dz2 + (1/rho)**2 * rho.dz * U.dz)
            else:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy)
                LapU = (1/rho * U.dx2 - (1/rho)**2 * rho.dx * U.dx +
                        1/rho * U.dy2 - (1/rho)**2 * rho.dy * U.dy)
        else:
            Lap = u.laplace
            LapU = U.laplace
            rho = 1
        # Derive stencils from symbolic equation
        first_eqn = m / rho * u.dt2 - Lap + damp * u.dt
        first_stencil = solve(first_eqn, u.forward)[0]
        second_eqn = m / rho * U.dt2 - LapU + damp * U.dt + dm * u.dt2
        second_stencil = solve(second_eqn, U.forward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: dt, h: model.get_spacing()}

        # Add Born-specific updates and resets
        stencils = [Eq(u.forward, first_stencil), Eq(U.forward, second_stencil)]
        super(BornOperator, self).__init__(nt, m.shape,
                                           stencils=stencils,
                                           subs=[subs, subs],
                                           spc_border=spc_order/2,
                                           time_order=time_order,
                                           forward=True,
                                           dtype=m.dtype,
                                           **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [dm, source, source.coordinates, rec, rec.coordinates, U]
        self.output_params = [rec]
        self.propagator.time_loop_stencils_b = source.add(m, u, t - 1)
        self.propagator.time_loop_stencils_a = rec.read(U)
        self.propagator.add_devito_param(dm)
        self.propagator.add_devito_param(source)
        self.propagator.add_devito_param(source.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)
        self.propagator.add_devito_param(U)


class ForwardOperatorD(Operator):
    def __init__(self, model, damp, data, qx, qy, qz=None,
                 time_order=2, spc_order=6, save=False, **kwargs):
        nt, nrec = data.traces.shape
        dt = model.get_critical_dt()
        u = TimeData(name="u", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order, save=save,
                     dtype=damp.dtype)
        u.pad_time = save
        s, h = symbols('s h')
        if len(model.get_shape_comp()) == 3:
            src_dipole = h * (qx.dx + qy.dy + qz.dz)
        else:
            src_dipole = h * (qx.dx + qy.dy)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * u.dx2 + (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 + (1/rho)**2 * rho.dy * u.dy +
                       1/rho * u.dz2 + (1/rho)**2 * rho.dz * u.dz)
            else:
                Lap = (1/rho * u.dx2 - (1/rho)**2 * rho.dx * u.dx +
                       1/rho * u.dy2 - (1/rho)**2 * rho.dy * u.dy)
        else:
            Lap = u.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * u.dt2 - Lap + damp * u.dt + src_dipole
        stencil = solve(eqn, u.forward)[0]
        # Add substitutions for spacing (temporal and spatial)
        subs = {s: dt, h: model.get_spacing()}
        super(ForwardOperatorD, self).__init__(nt, m.shape,
                                               stencils=Eq(u.forward, stencil),
                                               subs=subs,
                                               spc_border=spc_order/2,
                                               time_order=time_order,
                                               forward=True,
                                               dtype=m.dtype,
                                               **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [rec, rec.coordinates]
        self.output_params += [rec]
        self.propagator.time_loop_stencils_a = rec.read(u)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)


class AdjointOperatorD(Operator):
    def __init__(self, model, damp, data, recin,
                 time_order=2, spc_order=6, **kwargs):
        nrec, nt = data.traces.shape
        dt = model.get_critical_dt()
        v = TimeData(name="v", shape=model.get_shape_comp(), time_dim=nt,
                     time_order=time_order, space_order=spc_order,
                     save=False, dtype=damp.dtype)
        m = DenseData(name="m", shape=model.get_shape_comp(), dtype=damp.dtype)
        m.data[:] = model.padm()
        v.pad_time = False
        srca = SourceLike(name="srca", npoint=1, nt=nt, dt=dt, h=model.get_spacing(),
                          coordinates=np.array(data.source_coords,
                                               dtype=damp.dtype)[np.newaxis, :],
                          ndim=len(damp.shape), dtype=damp.dtype, nbpml=model.nbpml)
        rec = SourceLike(name="rec", npoint=nrec, nt=nt, dt=dt, h=model.get_spacing(),
                         coordinates=data.receiver_coords, ndim=len(damp.shape),
                         dtype=damp.dtype, nbpml=model.nbpml)
        rec.data[:] = recin[:]
        if model.rho is not None:
            rho = DenseData(name="rho", shape=model.get_shape_comp(),
                            dtype=damp.dtype, space_order=spc_order)
            rho.data[:] = model.pad(model.rho)
            if len(model.get_shape_comp()) == 3:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy +
                       1/rho * v.dz2 - (1/rho)**2 * rho.dz * v.dz)
            else:
                Lap = (1/rho * v.dx2 - (1/rho)**2 * rho.dx * v.dx +
                       1/rho * v.dy2 - (1/rho)**2 * rho.dy * v.dy)
        else:
            Lap = v.laplace
            rho = 1
        # Derive stencil from symbolic equation
        eqn = m / rho * v.dt2 - Lap - damp * v.dt
        stencil = solve(eqn, v.backward)[0]

        # Add substitutions for spacing (temporal and spatial)
        s, h = symbols('s h')
        subs = {s: model.get_critical_dt(), h: model.get_spacing()}
        super(AdjointOperatorD, self).__init__(nt, m.shape,
                                               stencils=Eq(v.backward, stencil),
                                               subs=subs,
                                               spc_border=spc_order/2,
                                               time_order=time_order,
                                               forward=False,
                                               dtype=m.dtype,
                                               **kwargs)

        # Insert source and receiver terms post-hoc
        self.input_params += [srca, srca.coordinates, rec, rec.coordinates]
        self.propagator.time_loop_stencils_a = rec.add(m, v) + srca.read(v)
        self.output_params = [srca]
        self.propagator.add_devito_param(srca)
        self.propagator.add_devito_param(srca.coordinates)
        self.propagator.add_devito_param(rec)
        self.propagator.add_devito_param(rec.coordinates)