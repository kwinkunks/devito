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
        obj = super(Temporary, cls).__new__(cls, lhs, rhs, **kwargs)
        obj._reads = set(reads)
        obj._readby = set(readby)
        return obj

    @property
    def reads(self):
        return self._reads

    @property
    def readby(self):
        return self._readby

    @property
    def is_time_invariant(self):
        # FIXME: temp1 = temp33*temp34 would be considered time-invariant, but
        # temp33 or temp34 could actually be time-dependent
        return t not in self.lhs.atoms() | self.rhs.atoms()

    @property
    def is_terminal(self):
        return len(self.readby) == 0

    def __repr__(self):
        return "DSE(%s, reads=%s, readby=%s)" % (super(Temporary, self).__repr__(),
                                                 str(self.reads), str(self.readby))


class Rewriter(object):

    """
    Transform expressions to create time-invariant computation.
    """

    # Do not rewrite expressions accessing more than THRESHOLD temporaries
    THRESHOLD = 20

    def __init__(self, expr, mode='basic'):
        self.expr = expr
        self.mode = mode

    def run(self):
        processed = self.expr

        if self.mode in ['basic', 'advanced']:
            time_invariants, processed = [], self._cse()

        if self.mode == 'advanced':
            graph = self._temporaries_graph(processed)
            time_invariants, processed = self._process_graph(graph)
            self._optimize_time_invariants(time_invariants, processed)

        return time_invariants + processed

    def _temporaries_graph(self, temporaries):
        """
        Create a dependency graph given a list of sympy.Eq.
        """
        mapper = OrderedDict()
        Node = namedtuple('Node', ['rhs', 'reads', 'readby'])

        for lhs, rhs in [i.args for i in temporaries]:
            mapper[lhs] = Node(rhs, {i for i in rhs.atoms() if i in mapper}, set())
            for i in mapper[lhs].reads:
                assert i in mapper, "Illegal Flow"
                mapper[i].readby.add(lhs)

        return [Temporary(lhs, node.rhs, reads=node.reads, readby=node.readby)
                for lhs, node in mapper.items()]

    def _process_graph(self, graph):
        """
        Extract time-invariant computation from a temporaries graph.
        """
        graph = OrderedDict([(i.lhs, i) for i in graph])
        processing = OrderedDict()

        time_invariants = OrderedDict()
        time_varying_syms = [i.lhs.base for i in self.expr]

        for lhs, node in graph.items():
            # Be sure we work on an up-to-date version of node
            node = processing.get(lhs, node)

            # Create time-invariant computation
            if not node.is_time_invariant:
                handle = expand_mul(node.rhs)

                indexed = handle.find(lambda i: isinstance(i, Indexed))
                collectable = [i for i in indexed if i.base in time_varying_syms]
                handle = collect(handle, collectable)

                start = len(time_invariants)
                reconstructed, mapper = self._create_time_invariants(handle, start)

                node = Temporary(lhs, reconstructed,
                                 reads=node.reads, readby=node.readby)

                for temp, value in mapper.items():
                    reads = {i for i in value.find(Indexed) if i in time_invariants}
                    time_invariants[temp] = Temporary(temp, value,
                                                      reads=reads, readby=[lhs])
            processing[lhs] = node

            # Substitute into subsequent temporaries
            # TODO : extend if to take THRESHOLD into account
            if not node.is_terminal:
                for j in node.readby:
                    handle = processing.get(j, graph[j])
                    reads = (handle.reads - {lhs}) | node.reads
                    processing[j] = Temporary(handle.lhs,
                                              handle.rhs.xreplace({lhs: node.rhs}),
                                              reads=reads, readby=handle.readby)
                processing.pop(lhs, None)

        # Reorder based on original position
        processing = sorted(processing.values(),
                            key=lambda n: graph.keys().index(n.lhs))

        return time_invariants.values(), processing

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
            # Return semantic: (reconstructed expr, time invariant flag)
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

    def _optimize_time_invariants(self, time_invariants, processed):
        """
        Eliminate duplicate time invariants and collect common factors.
        """
        # TODO
        pass

    def _cse(self):
        """
        Perform common subexpression elimination.
        """
        expr = self.expr if isinstance(self.expr, list) else [self.expr]

        temps, stencils = cse(expr, numbered_symbols("temp"))

        # Restores the LHS
        for i in range(len(expr)):
            stencils[i] = Eq(expr[i].lhs, stencils[i].rhs)

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
                        s_dict[arg] = Indexed(to_revert[value.base.label], *new_indices)
            to_revert[temp] = value.xreplace(s_dict)

        subs_dict = {}

        # Builds a dictionary of the replacements
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

        stencils = [stencil.xreplace(subs_dict) for stencil in stencils]

        to_keep = [Eq(temp[0], temp[1].xreplace(subs_dict)) for temp in to_keep]

        # If the RHS of a temporary variable is the LHS of a stencil,
        # update the value of the temporary variable after the stencil

        new_stencils = []

        for stencil in stencils:
            new_stencils.append(stencil)

            for temp in to_keep:
                if stencil.lhs in preorder_traversal(temp.rhs):
                    new_stencils.append(temp)
                    break

        return to_keep + new_stencils


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


def expression_shape(expr):
    indexed = set([e for e in preorder_traversal(expr) if isinstance(e, Indexed)])
    if not indexed:
        return ()
    indexed = sorted(indexed, key=lambda s: len(s.indices), reverse=True)
    indices = [flatten(j.free_symbols for j in i.indices) for i in indexed]
    assert all(set(indices[0]).issuperset(set(i)) for i in indices)
    return tuple(indices[0])
