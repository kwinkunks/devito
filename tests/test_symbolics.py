import numpy as np
from sympy import Add
import pytest

from devito.symbolics import Rewriter

from examples.tti.tti_example import setup
from examples.tti.tti_operators import ForwardOperator


@pytest.fixture(scope='module')
def operator_noopt():
    problem = setup(dimensions=(16, 16, 16), time_order=2, space_order=2, tn=10.0,
                    cse=False, auto_tuning=False, cache_blocking=None)
    operator = ForwardOperator(problem.model, problem.src, problem.damp,
                               problem.data, time_order=problem.t_order,
                               spc_order=problem.s_order, save=False,
                               cache_blocking=None, cse=False)
    return operator


@pytest.fixture(scope='module')
def operator_rewrite():
    problem = setup(dimensions=(16, 16, 16), time_order=2, space_order=2, tn=10.0,
                    cse=True, auto_tuning=False, cache_blocking=None)
    operator = ForwardOperator(problem.model, problem.src, problem.damp,
                               problem.data, time_order=problem.t_order,
                               spc_order=problem.s_order, save=False,
                               cache_blocking=None, cse=False)
    return operator


def test_tti_rewrite_output(operator_noopt):
    output1 = operator_noopt.apply()
    output2 = operator_noopt.apply()

    for o1, o2 in zip(output1, output2):
        assert np.isclose(np.linalg.norm(o1.data - o2.data), 0.0)


def test_tti_rewrite_temporaries_graph(operator_rewrite):
    op = operator_rewrite

    rewriter = Rewriter(op.stencils)

    processed = rewriter._cse()
    graph = rewriter._temporaries_graph(processed)

    assert len([v for v in graph.values() if v.is_terminal]) == len(op.stencils)
    assert len(graph) == len(processed)


@pytest.mark.parametrize('preprocess', [False, True])
def test_tti_rewrite_graph_splitting(operator_rewrite, preprocess):
    op = operator_rewrite

    rewriter = Rewriter(op.stencils)

    processed = rewriter._cse()
    graph = rewriter._temporaries_graph(processed)
    graph = rewriter._normalize_graph(graph)
    if preprocess:
        graph = rewriter._process_graph(graph)

    subgraphs = rewriter._split_into_subgraphs(graph)
    for stencil in op.stencils:
        stencil_subgraphs = [i.values() for i in subgraphs if stencil.lhs in i]
        terminals = []
        for subgraph in stencil_subgraphs:
            terminals.extend([i.rhs for i in subgraph if i.is_terminal])
        assert Add(*terminals) == graph[stencil.lhs].rhs
