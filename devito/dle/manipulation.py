from sympy import Eq

from devito.dle import is_foldable, retrieve_iteration_tree
from devito.nodes import Expression, Iteration, IterationFold, List
from devito.visitors import (FindAdjacentIterations, IsPerfectIteration,
                             MergeOuterIterations, NestedTransformer, Transformer)

__all__ = ['compose_nodes', 'copy_arrays', 'fold_iteration_tree',
           'unfold_iteration_tree']


def compose_nodes(nodes, retrieve=False):
    """
    Build an Iteration/Expression tree by nesting the nodes in ``nodes``.
    """
    l = list(nodes)
    tree = []

    body = l.pop(-1)
    while l:
        handle = l.pop(-1)
        body = handle._rebuild(body, **handle.args_frozen)
        tree.append(body)

    if retrieve is True:
        tree = list(reversed(tree))
        return body, tree
    else:
        return body


def copy_arrays(mapper, reverse=False):
    """
    Build an Iteration/Expression tree performing the copy ``k = v``, or
    ``v = k`` if reverse=True, for each (k, v) in mapper. (k, v) are expected
    to be of type :class:`IndexedData`. The loop bounds are inferred from
    the dimensions used in ``k``.
    """
    if not mapper:
        return ()

    # Build the Iteration tree for the copy
    iterations = []
    for k, v in mapper.items():
        handle = []
        indices = k.function.indices
        for i, j in zip(k.shape, indices):
            handle.append(Iteration([], dimension=j, limits=i))
        lhs, rhs = (v, k) if reverse else (k, v)
        handle.append(Expression(Eq(lhs[indices], rhs[indices]), dtype=k.function.dtype))
        iterations.append(compose_nodes(handle))

    # Maybe some Iterations are mergeable
    iterations = MergeOuterIterations().visit(iterations)

    return iterations


def fold_iteration_tree(node):
    """
    Create :class:`IterationFold`s from sequences of nested :class:`Iteration`.
    """
    found = FindAdjacentIterations().visit(node)
    found.pop('seen_iteration')

    mapper = {}
    for k, v in found.items():
        for i in v:
            # Check if the Iterations in /i/ are foldable or not
            assert len(i) > 1
            if any(not IsPerfectIteration().visit(j) for j in i):
                continue
            trees = [retrieve_iteration_tree(j)[0] for j in i]
            if any(len(trees[0]) != len(j) for j in trees):
                continue
            pairwise_folds = zip(*trees)
            if any(not is_foldable(j) for j in pairwise_folds):
                continue
            for j in pairwise_folds:
                root, remainder = j[0], j[1:]
                folds = [(tuple(x-y for x, y in zip(i.offsets, root.offsets)), i.nodes)
                         for i in remainder]
                mapper[root] = IterationFold(folds=folds, **root.args)
                for k in remainder:
                    mapper[k] = None

    # Insert the IterationFolds in the Iteration/Expression tree
    processed = NestedTransformer(mapper).visit(node)

    return processed


def unfold_iteration_tree(node):
    """
    Unfold nested :class:`IterationFold`.

    Examples
    ========
    Given a section of Iteration/Expression tree as below: ::

        for i = 1 to N-1  // folded
          for j = 1 to N-1  // folded
            foo1()

    Assuming a fold with offset 1 in both /i/ and /j/ and body ``foo2()``, create:

        for i = 1 to N-1
          for j = 1 to N-1
            foo1()
        for i = 2 to N-2
          for j = 2 to N-2
            foo2()
    """
    # Search the unfolding candidates
    candidates = []
    for tree in retrieve_iteration_tree(node):
        handle = tuple(i for i in tree if i.is_IterationFold)
        if handle:
            # Sanity check
            assert IsPerfectIteration().visit(handle[0])
            candidates.append(handle)

    # Perform unfolding
    mapper = {}
    for tree in candidates:
        unfolded = zip(*[i.unfold() for i in tree])
        unfolded = [compose_nodes(i) for i in unfolded]
        mapper[tree[0]] = List(body=unfolded)

    # Insert the unfolded Iterations in the Iteration/Expression tree
    processed = Transformer(mapper).visit(node)

    return processed
