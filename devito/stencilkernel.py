from __future__ import absolute_import

import operator
from collections import OrderedDict, defaultdict, namedtuple
from ctypes import c_double, c_int
from functools import reduce
from hashlib import sha1
from os import path

import cgen as c
import numpy as np

from devito.compiler import (get_compiler_from_env, get_tmp_dir,
                             jit_compile_and_load)
from devito.dimension import BufferedDimension, Dimension
from devito.dle import transform
from devito.dse import indexify, retrieve_and_check_dtype, rewrite
from devito.interfaces import SymbolicData
from devito.logger import error, info, warning
from devito.nodes import (Block, Expression, Function, Iteration,
                          TimedList, TypedExpression)
from devito.profiler import Profiler
from devito.tools import DefaultOrderedDict, as_tuple, filter_ordered, pprint
from devito.visitors import (EstimateCost, FindNodeType, FindSections, FindSymbols,
                             IsPerfectIteration, MergeOuterIterations,
                             ResolveIterationVariable, SubstituteExpression,
                             Transformer, printAST)

__all__ = ['StencilKernel']


class StencilKernel(Function):

    """
    Cache of auto-tuned StencilKernels.
    """
    _AT_cache = {}

    """A special :class:`Function` to evaluate stencils through just-in-time
    compilation of C code.

    :param stencils: SymPy equation or list of equations that define the
                     stencil used to create the kernel of this Operator.
    :param kwargs: Accept the following entries: ::

        * name : Name of the kernel function - defaults to "Kernel".
        * subs : Dict or list of dicts containing the SymPy symbol
                 substitutions for each stencil respectively.
        * dse : Use the Devito Symbolic Engine to optimize the expressions -
                defaults to "advanced".
        * dle : Use the Devito Loop Engine to optimize the loops -
                defaults to "advanced".
        * compiler: Compiler class used to perform JIT compilation.
                    If not provided, the compiler will be inferred from the
                    environment variable DEVITO_ARCH, or default to GNUCompiler.
        * profiler: :class:`devito.Profiler` instance to collect profiling
                    meta-data at runtime. Use profiler=None to disable profiling.
    """
    def __init__(self, stencils, **kwargs):
        name = kwargs.get("name", "Kernel")
        subs = kwargs.get("subs", {})
        dse = kwargs.get("dse", "advanced")
        dle = kwargs.get("dle", "advanced")
        compiler = kwargs.get("compiler", None)

        # Default attributes required for compilation
        self.compiler = compiler or get_compiler_from_env()
        self.profiler = kwargs.get("profiler", Profiler(self.compiler.openmp))
        self._includes = ['stdlib.h', 'math.h', 'sys/time.h']
        self._lib = None
        self._cfunction = None

        # Convert stencil expressions into actual Nodes, going through the
        # Devito Symbolic Engine for flop optimization.
        stencils = stencils if isinstance(stencils, list) else [stencils]
        dtype = retrieve_and_check_dtype(stencils)
        stencils = [indexify(s) for s in stencils]
        stencils = [s.xreplace(subs) for s in stencils]
        dse_state = rewrite(stencils, mode=dse)

        # Wrap expressions with Iterations according to dimensions
        nodes = self._schedule_expressions(dse_state, dtype)

        # Introduce C-level profiling infrastructure
        self.sections = OrderedDict()
        nodes = self._profile_sections(nodes)

        # Now resolve and substitute dimensions for loop index variables
        subs = {}
        nodes = ResolveIterationVariable().visit(nodes, subs=subs)
        nodes = SubstituteExpression(subs=subs).visit(nodes)

        # Apply the Devito Loop Engine for loop optimization and finalize instantiation
        dle_state = transform(nodes, mode=set_dle_mode(dle, self.compiler),
                              compiler=self.compiler)
        body = dle_state.nodes
        parameters = FindSymbols().visit(nodes)
        parameters += [i.argument for i in dle_state.arguments]
        super(StencilKernel, self).__init__(name, body, 'int', parameters, ())

        # DLE might have introduced additional headers
        self._includes.extend(list(dle_state.includes))

        # Track the DSE and DLE output, as they may be useful later
        self._dse_state = dse_state
        self._dle_state = dle_state

    def __call__(self, *args, **kwargs):
        self.apply(*args, **kwargs)

    def apply(self, *args, **kwargs):
        """Apply defined stencil kernel to a set of data objects"""
        if len(args) <= 0:
            args = self.parameters

        # Perform auto-tuning if the user requests it and loop blocking is in use
        maybe_autotune = kwargs.get('autotune', False)

        # Map of required arguments and actual dimension sizes
        arguments = OrderedDict([(arg.name, arg) for arg in self.parameters])
        dim_sizes = {}

        # Traverse positional args and infer loop sizes for open dimensions
        f_args = [f for f in arguments.values() if isinstance(f, SymbolicData)]
        for f, arg in zip(f_args, args):
            # Ensure we're dealing or deriving numpy arrays
            data = f.data if isinstance(f, SymbolicData) else arg
            if not isinstance(data, np.ndarray):
                error('No array data found for argument %s' % f.name)
            arguments[f.name] = data

            # Ensure data dimensions match symbol dimensions
            for i, dim in enumerate(f.indices):
                # Infer open loop limits
                if dim.size is None:
                    if isinstance(dim, BufferedDimension):
                        # Check if provided as a keyword arg
                        size = kwargs.get(dim.name, None)
                        if size is None:
                            error("Unknown dimension size, please provide "
                                  "size via Kernel.apply(%s=<size>)" % dim.name)
                            raise RuntimeError('Dimension of unspecified size')
                        dim_sizes[dim] = size
                    elif dim in dim_sizes:
                        # Ensure size matches previously defined size
                        assert dim_sizes[dim] == data.shape[i]
                    else:
                        # Derive size from grid data shape and store
                        dim_sizes[dim] = data.shape[i]
                else:
                    if not isinstance(dim, BufferedDimension):
                        assert dim.size == data.shape[i]

        # Add user-provided block sizes, if any
        dle_arguments = OrderedDict()
        for i in self._dle_state.arguments:
            dim_size = dim_sizes.get(i.original_dim, i.original_dim.size)
            assert dim_size is not None, "Unable to match arguments and values"
            if i.value:
                try:
                    dle_arguments[i.argument] = i.value(dim_size)
                except TypeError:
                    dle_arguments[i.argument] = i.value
                    # User-provided block size available, do not autotune
                    maybe_autotune = False
            else:
                dle_arguments[i.argument] = dim_size
        dim_sizes.update(dle_arguments)

        # Insert loop size arguments from dimension values
        d_args = [d for d in arguments.values() if isinstance(d, Dimension)]
        for d in d_args:
            arguments[d.name] = dim_sizes[d]

        # Retrieve the data type of the arrays
        try:
            dtypes = [i.data.dtype for i in f_args]
            dtype = dtypes[0]
            if any(i != dtype for i in dtypes):
                warning("Found non-matching data types amongst the provided"
                        "symbolic arguments.")
            dtype_size = dtype.itemsize
        except IndexError:
            dtype_size = 1

        # Might have been asked to auto-tune the block size
        if maybe_autotune:
            self._autotune(arguments)

        # Add profiler structs
        if self.profiler:
            cpointer = self.profiler.as_ctypes_pointer(Profiler.TIME)
            arguments[self.profiler.s_name] = cpointer

        # Invoke kernel function with args
        self.cfunction(*list(arguments.values()))

        # Summary of performance achieved
        info("="*79)
        for itspace, profile in self.sections.items():
            # Time
            elapsed = self.profiler.timings[profile.timer]
            # Flops
            niters = reduce(operator.mul, [i.size or dim_sizes[i] for i in itspace])
            flops = float(profile.ops*niters)
            gflops = flops/10**9
            # Compulsory traffic
            traffic = profile.memory*niters*dtype_size

            info("Section %s with OI=%.2f computed in %.2f s [Perf: %.2f GFlops/s]" %
                 (str(itspace), flops/traffic, elapsed, gflops/elapsed))
        info("="*79)

    def _profile_sections(self, nodes):
        """Introduce C-level profiling nodes within the Iteration/Expression tree."""
        mapper = {}
        for i, expr in enumerate(nodes):
            for itspace in FindSections().visit(expr).keys():
                for j in itspace:
                    if IsPerfectIteration().visit(j) and j not in mapper:
                        # Insert `TimedList` block. This should come from
                        # the profiler, but we do this manually for now.
                        lname = 'loop_%s_%d' % (j.index, i)
                        mapper[j] = TimedList(gname=self.profiler.t_name,
                                              lname=lname, body=j)
                        self.profiler.t_fields += [(lname, c_double)]

                        # Estimate computational properties of the timed section
                        # (operational intensity, memory accesses)
                        k = tuple(k.dim for k in itspace)
                        v = EstimateCost().visit(j)
                        self.sections[k] = Profile(lname, v.ops, v.mem)
                        break
        processed = [Transformer(mapper).visit(Block(body=nodes))]
        return processed

    def _autotune(self, arguments):
        """Use auto-tuning on this StencilKernel to determine empirically the
        best block sizes (when loop blocking is in use). The block sizes tested
        are those listed in ``options['at_blocksizes']``, plus the case that is
        as if blocking were not applied (ie, unitary block size)."""

        at_arguments = arguments.copy()

        # Output data must not be changed
        output = [i.base.label.name for i in self._dse_state.output_fields]
        for k, v in arguments.items():
            if k in output:
                at_arguments[k] = v.copy()

        # Squeeze dimensions to minimize auto-tuning time
        iterations = FindNodeType(Iteration).visit(self.body)
        squeezable = [i.dim.name for i in iterations if 'sequential' in i.properties]

        # Attempted block sizes
        mapper = OrderedDict([(i.argument.name, i) for i in self._dle_state.arguments])
        blocksizes = [OrderedDict([(i, v) for i in mapper])
                      for v in options['at_blocksize']]
        blocksizes += [OrderedDict([(k, 1) for k, v in mapper.items()])]

        # Note: there is only a single loop over 'blocksize' because only
        # square blocks are tested
        timings = OrderedDict()
        for blocksize in blocksizes:
            illegal = False
            for k, v in at_arguments.items():
                if k in blocksize:
                    val = blocksize[k]
                    handle = at_arguments.get(mapper[k].original_dim.name)
                    if val <= mapper[k].iteration.end(handle):
                        at_arguments[k] = val
                    else:
                        # Block size cannot be larger than actual dimension
                        illegal = True
                        break
                elif k in squeezable:
                    at_arguments[k] = options['at_squeezer']
            if illegal:
                continue

            # Add profiler structs
            if self.profiler:
                cpointer = self.profiler.as_ctypes_pointer(Profiler.TIME)
                at_arguments[self.profiler.s_name] = cpointer

            self.cfunction(*list(at_arguments.values()))
            timings[tuple(blocksize.items())] = sum(self.profiler.timings.values())

        best = dict(min(timings, key=timings.get))
        for k, v in arguments.items():
            if k in mapper:
                arguments[k] = best[k]

        info('Auto-tuned block shape: %s' % best)

    def _schedule_expressions(self, dse_state, dtype):
        """Wrap :class:`Expression` objects within suitable hierarchies of
        :class:`Iteration` according to dimensions.
        """
        processed = []
        for cluster in dse_state.clusters:
            # Build declarations or assignments
            body = []
            for k, v in cluster.items():
                if cluster.is_index(k):
                    body.append(TypedExpression(v, np.int32))
                elif v.is_terminal:
                    body.append(Expression(v))
                else:
                    body.append(TypedExpression(v, dtype))
            offsets = body[-1].index_offsets
            # Filter out aliasing due to buffered dimensions
            key = lambda d: d.parent if d.is_Buffered else d
            dimensions = filter_ordered(list(offsets.keys()), key=key)
            for d in reversed(dimensions):
                body = Iteration(body, dimension=d, limits=d.size, offsets=offsets[d])
            processed.append(body)

        # Merge Iterations iff outermost iterations agree
        processed = MergeOuterIterations().visit(processed)

        # Remove temporaries became redundat after squashing Iterations
        mapper = {}
        for k, v in FindSections().visit(processed).items():
            found = set()
            newexprs = []
            for n in v:
                newexprs.extend([n] if n.stencil not in found else [])
                found.add(n.stencil)
            mapper[k[-1]] = Iteration(newexprs, **k[-1].args_frozen)
        processed = Transformer(mapper).visit(processed)

        return processed

    @property
    def _cparameters(self):
        cparameters = super(StencilKernel, self)._cparameters
        cparameters += [c.Pointer(c.Value('struct %s' % self.profiler.s_name,
                                          self.profiler.t_name))]
        return cparameters

    @property
    def ccode(self):
        """Returns the C code generated by this kernel.

        This function generates the internal code block from Iteration
        and Expression objects, and adds the necessary template code
        around it.
        """
        blankline = c.Line("")

        # Generate function body with all the trimmings
        body = [e.ccode for e in self.body]
        ret = [c.Statement("return 0")]
        kernel = c.FunctionBody(self._ctop, c.Block(self._ccasts + body + ret))

        # Generate elemental functions produced by the DLE
        elemental_functions = [e.ccode for e in self._dle_state.elemental_functions]
        elemental_functions += [blankline]

        # Generate file header with includes and definitions
        includes = [c.Include(i, system=False) for i in self._includes]
        includes += [blankline]
        profiling = [self.profiler.as_cgen_struct(Profiler.TIME), blankline]
        return c.Module(includes + profiling + elemental_functions + [kernel])

    @property
    def basename(self):
        """Generate the file basename path for auto-generated files

        The basename is generated from the hash string of the kernel,
        which is base on the final expressions and iteration symbols.

        :returns: The basename path as a string
        """
        expr_string = printAST(self.body, verbose=True)
        expr_string += printAST(self._dle_state.elemental_functions, verbose=True)
        hash_key = sha1(expr_string.encode()).hexdigest()

        return path.join(get_tmp_dir(), hash_key)

    @property
    def cfunction(self):
        """Returns the JIT-compiled C function as a ctypes.FuncPtr object

        Note that this invokes the JIT compilation toolchain with the
        compiler class derived in the constructor

        :returns: The generated C function
        """
        if self._lib is None:
            self._lib = jit_compile_and_load(self.ccode, self.basename,
                                             self.compiler)
        if self._cfunction is None:
            self._cfunction = getattr(self._lib, self.name)
            self._cfunction.argtypes = self.argtypes

        return self._cfunction

    @property
    def argtypes(self):
        """Create argument types for defining function signatures via ctypes

        :returns: A list of ctypes of the matrix parameters and scalar parameters
        """
        return [c_int if isinstance(v, Dimension) else
                np.ctypeslib.ndpointer(dtype=v.dtype, flags='C')
                for v in self.parameters]


"""
A dict of standard names to be used for code generation
"""
cnames = {
    'loc_timer': 'loc_timer',
    'glb_timer': 'glb_timer'
}

"""
StencilKernel options
"""
options = {
    'at_squeezer': 3,
    'at_blocksize': [8, 16, 32]
}

"""
A helper to track profiled sections of code.
"""
Profile = namedtuple('Profile', 'timer ops memory')


def set_dle_mode(mode, compiler):
    """
    Transform :class:`StencilKernel` input in a format understandable by the DLE.
    """
    if not mode:
        return 'noop'
    mode = as_tuple(mode)
    params = mode[-1]
    if isinstance(params, dict):
        params['openmp'] = compiler.openmp
    else:
        params = {'openmp': compiler.openmp}
        mode += (params,)
    return mode
