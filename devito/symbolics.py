"""
The Devito symbolic engine is built on top of SymPy and provides two
classes of functions:
- for inspection of expressions
- for (in-place) manipulation of expressions
- for creation of new objects given some expressions
All exposed functions are prefixed with 'dse' (devito symbolic engine)
"""

from collections import namedtuple, OrderedDict

import numpy as np

from sympy import *

from devito.dimension import t, x, y, z
from devito.interfaces import SymbolicData
from devito.logger import warning

__all__ = ['dse_dimensions', 'dse_symbols', 'dse_dtype', 'dse_indexify',
           'dse_rewrite', 'dse_tolambda']

_temp_prefix = 'temp'


# Inspection

def dse_dimensions(expr):
    """
    Collect all function dimensions used in a sympy expression.
    """
    dimensions = []

    for e in preorder_traversal(expr):
        if isinstance(e, SymbolicData):
            dimensions += [i for i in e.indices if i not in dimensions]

    return dimensions


def dse_symbols(expr):
    """
    Collect defined and undefined symbols used in a sympy expression.

    Defined symbols are functions that have an associated :class
    SymbolicData: object, or dimensions that are known to the devito
    engine. Undefined symbols are generic `sympy.Function` or
    `sympy.Symbol` objects that need to be substituted before generating
    operator C code.
    """
    defined = set()
    undefined = set()

    for e in preorder_traversal(expr):
        if isinstance(e, SymbolicData):
            defined.add(e.func(*e.indices))
        elif isinstance(e, Function):
            undefined.add(e)
        elif isinstance(e, Symbol):
            undefined.add(e)

    return list(defined), list(undefined)


def dse_dtype(expr):
    """
    Try to infer the data type of an expression.
    """
    dtypes = [e.dtype for e in preorder_traversal(expr) if hasattr(e, 'dtype')]
    return np.find_common_type(dtypes, [])


# Manipulation

def dse_indexify(expr):
    """
    Convert functions into indexed matrix accesses in sympy expression.

    :param expr: sympy function expression to be converted.
    """
    replacements = {}

    for e in preorder_traversal(expr):
        if hasattr(e, 'indexed'):
            replacements[e] = e.indexify()

    return expr.xreplace(replacements)


def dse_rewrite(expr, mode='basic'):
    """
    Transform expressions to create time-invariant computation.
    """
    assert mode in ['no-op', 'basic', 'advanced']
    return Rewriter(expr, mode).run()


class Temporary(Eq):

    """
    A special sympy.Eq which can keep track of other sympy.Eq depending on self.
    """

    def __new__(cls, lhs, rhs, **kwargs):
        reads = kwargs.pop('reads', [])
        readby = kwargs.pop('readby', [])
        time_invariant = kwargs.pop('time_invariant', False)
        scope = kwargs.pop('scope', 0)
        obj = super(Temporary, cls).__new__(cls, lhs, rhs, **kwargs)
        obj._reads = set(reads)
        obj._readby = set(readby)
        obj._is_time_invariant = time_invariant
        obj._scope = scope
        obj._hoistable = False
        return obj

    @property
    def reads(self):
        return self._reads

    @property
    def readby(self):
        return self._readby

    @property
    def is_time_invariant(self):
        return self._is_time_invariant

    @property
    def is_terminal(self):
        return len(self.readby) == 0

    @property
    def is_tensor(self):
        return isinstance(self.lhs, Indexed) and self.lhs.rank > 0

    @property
    def is_scalarizable(self):
        return self.is_time_invariant and not self.is_terminal and self.is_tensor

    @property
    def scope(self):
        return self._scope

    @property
    def inflow(self):
        return len(self._reads)

    @property
    def outflow(self):
        return len(self._readby)

    @property
    def hoistable(self):
        return self._hoistable

    def try_make_hoistable(self):
        if self.is_time_invariant:
            self._hoistable = True

    def __repr__(self):
        return "DSE(%s, reads=%s, readby=%s)" % (super(Temporary, self).__repr__(),
                                                 str(self.reads), str(self.readby))


class Trace(OrderedDict):

    def __init__(self, root, graph, *args, **kwargs):
        super(Trace, self).__init__(*args, **kwargs)
        self._root = root
        self._compute(graph)

    def _compute(self, graph):
        if self.root not in graph:
            return
        to_visit = [(graph[self.root], 0)]
        while to_visit:
            temporary, level = to_visit.pop(0)
            self.__setitem__(temporary.lhs, level)
            to_visit.extend([(graph[i], level + 1) for i in temporary.reads])

    @property
    def root(self):
        return self._root

    @property
    def length(self):
        return len(self)

    def intersect(self, other):
        return Trace(self.root, {}, [(k, v) for k, v in self.items() if k in other])

    def union(self, other):
        return Trace(self.root, {}, [(k, v) for k, v in self.items() + other.items()])


class Rewriter(object):

    """
    Transform expressions to create time-invariant computation.
    """

    MAX_GRAPH_SIZE = 20

    def __init__(self, expr, mode='basic'):
        self.expr = expr
        self.mode = mode

    def run(self):
        processed = self.expr

        if self.mode in ['basic', 'advanced']:
            processed = self._cse(processed)

        if self.mode == 'advanced':
            graph = self._temporaries_graph(processed)
            graph = self._normalize_graph(graph)
            graph = self._process_graph(graph)
            subgraphs = self._split_into_subgraphs(graph)
            subgraphs = self._optimize_graph(subgraphs)
            processed = flatten(i.values() for i in subgraphs)

        return processed

    def _temporaries_graph(self, temporaries, scope=0):
        """
        Create a dependency graph given a list of sympy.Eq.
        """
        mapper = OrderedDict()
        Node = namedtuple('Node', ['rhs', 'reads', 'readby', 'time_invariant'])

        for lhs, rhs in [i.args for i in temporaries]:
            reads = {i for i in terminals(rhs) if i in mapper}
            mapper[lhs] = Node(rhs, reads, set(), is_time_invariant(rhs, mapper))
            for i in mapper[lhs].reads:
                assert i in mapper, "Illegal Flow"
                mapper[i].readby.add(lhs)

        nodes = [Temporary(k, v.rhs, reads=v.reads, readby=v.readby,
                           time_invariant=v.time_invariant, scope=scope)
                 for k, v in mapper.items()]
        return OrderedDict([(i.lhs, i) for i in nodes])

    def _normalize_graph(self, graph):
        """
        Reduce terminals to a sum-of-muls form.
        """
        normalized = []

        for k, v in graph.items():
            if v.is_terminal and not v.rhs.is_Add:
                normalized.append(Eq(k, expand_mul(v.rhs)))
            else:
                normalized.append(v)

        return self._temporaries_graph(normalized)

    def _split_into_subgraphs(self, graph):
        """
        Split a temporaries graph into multiple, smaller subgraphs. A heuristic
        typical of graph coloring algorithms is used: nodes that, in a given
        scheduling step, have maximal sharing are put in the same subgraph. No
        more than self.MAX_GRAPH_SIZE nodes can stay in a subgraph.
        """
        subgraphs = []
        terminals = [i for i in graph.values() if i.is_terminal]
        traces = {i: Trace(i, graph) for i in graph.keys()}

        for terminal in terminals:
            subgraph, args, index = OrderedDict(), [], 0
            schedule = list(terminal.rhs.args)
            while schedule:
                handle = schedule.pop(index)
                args.append(handle)
                trace = trace_union(handle.args, traces)
                for k, v in reversed(trace.items()):
                    subgraph[k] = graph[k]
                if not schedule or len(subgraph) > Rewriter.MAX_GRAPH_SIZE:
                    subgraph = subgraph.values() + [Eq(terminal.lhs, Add(*args))]
                    subgraph = self._temporaries_graph(subgraph, len(subgraphs))
                    subgraphs.append(subgraph)
                    subgraph, args, index = OrderedDict(), [], 0
                else:
                    scores = [len(trace.intersect(trace_union(i.args, traces)))
                              for i in schedule]
                    index = scores.index(max(scores))

        return subgraphs

    def _process_graph(self, graph):
        """
        Extract time-invariant computation from a temporaries graph.
        """
        processing = OrderedDict()
        time_invariants = OrderedDict()
        time_varying_syms = [i.lhs.base for i in self.expr]

        for lhs, node in graph.items():
            # Be sure we work on an up-to-date version of node
            node = processing.get(lhs, node)

            # Create time-invariant computation
            if not is_time_invariant(node):
                handle = expand_mul(node.rhs)

                # SymPy's collect is insanely slow, so we handcraft our own
                # factorization. Note: after expansion handle is in sum-of-mul form
                if handle.is_Add:
                    mapper, others = {}, []
                    for arg in handle.args:
                        factorizable = [i for i in arg.args if isinstance(i, Indexed)
                                        and i.base in time_varying_syms]
                        if len(factorizable) == 1:
                            mapper.setdefault(factorizable[0], []).append(arg)
                        else:
                            others.append(arg)
                    factorized = [Add(*v).collect(k) for k, v in mapper.items()]
                    handle = Add(*(factorized + others))

                start = len(time_invariants)
                rebuilt, mapper = self._create_time_invariants(handle, start)

                # Can I reuse some temporaries ?
                reads = []
                for k, v in mapper.items():
                    if v in time_invariants.values():
                        index = time_invariants.values().index(v)
                        found = time_invariants.keys()[index]
                        rebuilt = rebuilt.xreplace({k: found})
                        reads.append(found)
                    else:
                        time_invariants[k] = v
                        reads.append(k)
                reads = set(reads) or node.reads

                node = Temporary(lhs, rebuilt, reads=reads, readby=node.readby)

            processing[lhs] = node

            # Substitute into subsequent temporaries
            if not node.is_terminal:
                for j in node.readby:
                    handle = processing.get(j, graph[j])
                    reads = (handle.reads - {lhs}) | node.reads
                    processing[j] = Temporary(handle.lhs,
                                              handle.rhs.xreplace({lhs: node.rhs}),
                                              reads=reads, readby=handle.readby)
                processing.pop(lhs, None)

        processed = [Eq(k, v) for k, v in time_invariants.items()] +\
            processing.values()

        return self._temporaries_graph(processed)

    def _optimize_graph(self, graphs):
        """
        Apply the following transformations to each item in graphs: ::

            * Heuristic collection of common factors.
            * Contraction to scalars (e.g., t[i][j] -> t)
            * Common sub-expressions elimination
            * TODO: Cost-model driven time invariants declaration

        :param graphs: a single graph or a list of graphs.
        """
        graphs = [graphs] if isinstance(graphs, dict) else graphs

        # TODO: Heuristic collection of common factors
        optimized_graphs = []
        for graph in graphs:
            optimized_graphs.append(graph)

        # Contraction to scalars + CSE
        graphs, optimized_graphs = optimized_graphs, []
        mapper = {}
        for graph in graphs:
            assert all(graph.values()[0].scope == v.scope for v in graph.values())
            scope = graph.values()[0].scope
            contracted = []
            for k, v in graph.items():
                mapper[k] = k.base.label if v.is_scalarizable else k
                contracted.append(Eq(k, v.rhs).xreplace(mapper))
            contracted = self._cse(contracted)
            optimized_graphs.append(self._temporaries_graph(contracted, scope))

        # TODO: Time invariants declaration
        for graph in optimized_graphs:
            for k, v in graph.items():
                v.try_make_hoistable()

        return optimized_graphs

    def _create_time_invariants(self, expr, start=0):
        """
        Create a new expr' given expr where the longest time-invariant
        sub-expressions are replaced by temporaries. A mapper from the
        introduced temporaries to the corresponding time-invariant
        sub-expressions is also returned.

        Examples
        ========

        (a+b)*c[t] + s*d[t] + v*(d + e[t] + r)
            --> (t1*c[t] + s*d[t] + v*(e[t] + t2), {t1: (a+b), t2: (d+r)})
        (a*b[t] + c*d[t])*v[t]
            --> ((a*b[t] + c*d[t])*v[t], {})
        """

        def run(expr, root, mapper):
            # Return semantic: (rebuilt expr, time invariant flag)
            if expr in [S.Zero, S.One, S.NegativeOne, S.Half]:
                return (expr.func(), True)
            elif expr.is_Atom:
                return (expr.func(*expr.atoms()), True)
            elif isinstance(expr, Indexed):
                return (expr.func(*expr.args), t not in expr.atoms())
            else:
                children = [run(a, root, mapper) for a in expr.args]
                invariants = [a for a, flag in children if flag]
                varying = [a for a, _ in children if a not in invariants]
                if not invariants:
                    # Nothing is time-invariant
                    return (expr.func(*varying), False)
                if len(invariants) == len(children):
                    # Everything is time-invariant
                    if expr == root:
                        # Root is a special case
                        base = '%s_ti_%d' % (_temp_prefix, len(mapper)+start)
                        temporary = Indexed(base, *expression_shape(expr))
                        mapper[temporary] = expr.func(*expr.args)
                        return (temporary, True)
                    else:
                        # Go look for longer expressions first
                        return (expr.func(*invariants), True)
                else:
                    # Some children are time-invariant, but expr is time-dependent
                    if len(invariants) == 1 and \
                            isinstance(invariants[0], (Atom, Indexed)):
                        return (expr.func(*(invariants + varying)), False)
                    else:
                        base = '%s_ti_%d' % (_temp_prefix, len(mapper)+start)
                        shapes = [expression_shape(a) for a in invariants]
                        shapes = [i for i in shapes if i]
                        assert all(shapes[0] == i for i in shapes)
                        temporary = Indexed(base, *shapes[0])
                        mapper[temporary] = expr.func(*invariants)
                        return (expr.func(*(varying + [temporary])), False)

        mapper = OrderedDict()
        return run(expr, expr, mapper)[0], mapper

    def _cse(self, exprs=None):
        """
        Perform common subexpression elimination.
        """
        if exprs is None:
            exprs = self.expr
        if not isinstance(exprs, list):
            exprs = [exprs]

        temps, stencils = cse(exprs, numbered_symbols("temp"))

        # Restores the LHS
        for i in range(len(exprs)):
            stencils[i] = Eq(exprs[i].lhs, stencils[i].rhs)

        to_revert = {}
        to_keep = []

        # Restores IndexedBases if they are collected by CSE and
        # reverts changes to simple index operations (eg: t - 1)
        for temp, value in temps:
            if isinstance(value, IndexedBase):
                to_revert[temp] = value
            elif isinstance(value, Indexed):
                to_revert[temp] = value
            elif isinstance(value, Add) and not \
                    set([t, x, y, z]).isdisjoint(set(value.args)):
                to_revert[temp] = value
            else:
                to_keep.append((temp, value))

        # Restores the IndexedBases and the Indexes in the assignments to revert
        for temp, value in to_revert.items():
            s_dict = {}
            for arg in preorder_traversal(value):
                if isinstance(arg, Indexed):
                    new_indices = []
                    for index in arg.indices:
                        if index in to_revert:
                            new_indices.append(to_revert[index])
                        else:
                            new_indices.append(index)
                    if arg.base.label in to_revert:
                        s_dict[arg] = Indexed(to_revert[value.base.label],
                                              *new_indices)
            to_revert[temp] = value.xreplace(s_dict)

        # Builds a dictionary of the replacements
        subs_dict = {}
        for expr in stencils + [assign for temp, assign in to_keep]:
            for arg in preorder_traversal(expr):
                if isinstance(arg, Indexed):
                    new_indices = []
                    for index in arg.indices:
                        if index in to_revert:
                            new_indices.append(to_revert[index])
                        else:
                            new_indices.append(index)
                    if arg.base.label in to_revert:
                        subs_dict[arg] = Indexed(to_revert[arg.base.label], *new_indices)
                    elif tuple(new_indices) != arg.indices:
                        subs_dict[arg] = Indexed(arg.base, *new_indices)
                if arg in to_revert:
                    subs_dict[arg] = to_revert[arg]

        TMP = stencils
        restored_stencils, restored_temps = [], []
        for stencil in stencils:
            old, new = stencil, stencil.xreplace(subs_dict)
            while new != old:
                old, new = new, new.xreplace(subs_dict)
            restored_stencils.append(new)
        for temp, assign in to_keep: 
            old, new = assign, assign.xreplace(subs_dict)
            while new != old:
                old, new = new, new.xreplace(subs_dict)
            restored_temps.append(Eq(temp, new))
        stencils = restored_stencils
        to_keep = restored_temps

        # If the RHS of a temporary variable is the LHS of a stencil,
        # update the value of the temporary variable after the stencil
        new_stencils = []
        for stencil in stencils:
            new_stencils.append(stencil)

            for temp in to_keep:
                if stencil.lhs in preorder_traversal(temp.rhs):
                    new_stencils.append(temp)
                    break

        # Reshuffle to make sure temporaries come later than their read values
        processed = OrderedDict([(i.lhs, i) for i in to_keep + new_stencils])
        temporaries = set(processed.keys())
        ordered = OrderedDict()
        while processed:
            k, v = processed.popitem(last=False)
            temporary_reads = terminals(v.rhs) & temporaries
            if all(i in ordered for i in temporary_reads):
                ordered[k] = v
            else:
                # Must wait for some earlier temporaries, push back into queue
                processed[k] = v

        return ordered.values()


# Creation

def dse_tolambda(exprs):
    """
    Tranform an expression into a lambda.

    :param exprs: an expression or a list of expressions.
    """
    exprs = exprs if isinstance(exprs, list) else [exprs]

    lambdas = []

    for expr in exprs:
        terms = free_terms(expr.rhs)
        term_symbols = [symbols("i%d" % i) for i in range(len(terms))]

        # Substitute IndexedBase references to simple variables
        # lambdify doesn't support IndexedBase references in expressions
        tolambdify = expr.rhs.subs(dict(zip(terms, term_symbols)))
        lambdified = lambdify(term_symbols, tolambdify)
        lambdas.append((lambdified, terms))

    return lambdas


# Utilities

def free_terms(expr):
    """
    Find the free terms in an expression.
    """
    found = []

    for term in expr.args:
        if isinstance(term, Indexed):
            found.append(term)
        else:
            found += free_terms(term)

    return found


def is_time_invariant(expr, mapper=None):
    """
    Check if expr is time invariant. A mapper from symbols to values may be
    provided to determine whether any of the symbols involved in the evaluation
    of expr are time-dependent. If a symbol in expr does not appear in mapper,
    then time invariance will be inferred from its shape.
    """
    is_time_dependent = lambda e: t in e.atoms()
    mapper = mapper or {}

    if expr.is_Equality:
        if is_time_dependent(expr.lhs):
            return False
        else:
            expr = expr.rhs

    to_visit = terminals(expr)
    while to_visit:
        symbol = to_visit.pop()
        if is_time_dependent(symbol):
            return False
        if symbol in mapper:
            to_visit |= terminals(mapper[symbol].rhs)

    return True


def terminals(expr):
    indexed = list(expr.find(Indexed))

    # To be discarded
    junk = flatten(i.atoms() for i in indexed)

    symbols = list(expr.find(Symbol))
    symbols = [i for i in symbols if i not in junk]

    return set(indexed + symbols)


def trace_union(iterable, mapper):
    """
    Compute the union of the traces of the symbols in iterable retrieved from mapper.
    """
    in_trace = [i for i in iterable if i in mapper]
    if in_trace:
        handle = mapper[in_trace.pop(0)]
    else:
        handle = Trace(None, mapper)
    while in_trace:
        handle = handle.union(mapper[in_trace.pop(0)])
    return handle


def expression_shape(expr):
    indexed = set([e for e in preorder_traversal(expr) if isinstance(e, Indexed)])
    if not indexed:
        return ()
    indexed = sorted(indexed, key=lambda s: len(s.indices), reverse=True)
    indices = [flatten(j.free_symbols for j in i.indices) for i in indexed]
    assert all(set(indices[0]).issuperset(set(i)) for i in indices)
    return tuple(indices[0])


def estimate_cost(handle):
    try:
        # Is it a plain SymPy object ?
        iter(handle)
    except TypeError:
        handle = [handle]
    try:
        # Is it a dict ?
        handle = handle.values()
    except AttributeError:
        try:
            # Must be a list of dicts then
            handle = flatten(i.values() for i in handle)
        except AttributeError:
            pass
    # At this point it must be a list of SymPy objects
    try:
        return sum(count_ops(i) for i in handle)
    except:
        warning("Cannot estimate cost of %s" % str(handle))
