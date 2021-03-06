from __future__ import absolute_import

from functools import reduce
from operator import mul
import numpy as np
import pytest
from sympy import Eq

from devito.dle import transform
from devito.dle.backends import DevitoRewriter as Rewriter
from devito.interfaces import DenseData, TimeData
from devito.nodes import Expression, Function, List
from devito.operator import Operator
from devito.visitors import ResolveIterationVariable, SubstituteExpression


@pytest.fixture(scope="module")
def exprs(a, b, c, d, a_dense, b_dense):
    return [Expression(Eq(a, a + b + 5.)),
            Expression(Eq(a, b*d - a*c)),
            Expression(Eq(b, a + b*b + 3)),
            Expression(Eq(a, a*b*d*c)),
            Expression(Eq(a, 4 * ((b + d) * (a + c)))),
            Expression(Eq(a, (6. / b) + (8. * a))),
            Expression(Eq(a_dense, a_dense + b_dense + 5.))]


@pytest.fixture(scope="module")
def simple_function(a, b, c, d, exprs, iters):
    # void foo(a, b)
    #   for i
    #     for j
    #       for k
    #         expr0
    #         expr1
    symbols = [i.base.function for i in [a, b, c, d]]
    body = iters[0](iters[1](iters[2]([exprs[0], exprs[1]])))
    f = Function('foo', body, 'void', symbols, ())
    subs = {}
    f = ResolveIterationVariable().visit(f, subs=subs)
    f = SubstituteExpression(subs=subs).visit(f)
    return f


@pytest.fixture(scope="module")
def simple_function_with_paddable_arrays(a_dense, b_dense, exprs, iters):
    # void foo(a_dense, b_dense)
    #   for i
    #     for j
    #       for k
    #         expr0
    symbols = [i.base.function for i in [a_dense, b_dense]]
    body = iters[0](iters[1](iters[2](exprs[6])))
    f = Function('foo', body, 'void', symbols, ())
    subs = {}
    f = ResolveIterationVariable().visit(f, subs=subs)
    f = SubstituteExpression(subs=subs).visit(f)
    return f


@pytest.fixture(scope="module")
def simple_function_fissionable(a, b, exprs, iters):
    # void foo(a, b)
    #   for i
    #     for j
    #       for k
    #         expr0
    #         expr2
    symbols = [i.base.function for i in [a, b]]
    body = iters[0](iters[1](iters[2]([exprs[0], exprs[2]])))
    f = Function('foo', body, 'void', symbols, ())
    subs = {}
    f = ResolveIterationVariable().visit(f, subs=subs)
    f = SubstituteExpression(subs=subs).visit(f)
    return f


@pytest.fixture(scope="module")
def complex_function(a, b, c, d, exprs, iters):
    # void foo(a, b, c, d)
    #   for i
    #     for s
    #       expr0
    #     for j
    #       for k
    #         expr1
    #         expr2
    #     for p
    #       expr3
    symbols = [i.base.function for i in [a, b, c, d]]
    body = iters[0]([iters[3](exprs[2]),
                     iters[1](iters[2]([exprs[3], exprs[4]])),
                     iters[4](exprs[5])])
    f = Function('foo', body, 'void', symbols, ())
    subs = {}
    f = ResolveIterationVariable().visit(f, subs=subs)
    f = SubstituteExpression(subs=subs).visit(f)
    return f


def _new_operator1(shape, **kwargs):
    infield = DenseData(name='in', shape=shape, dtype=np.int32)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)

    outfield = DenseData(name='out', shape=shape, dtype=np.int32)

    stencil = Eq(outfield.indexify(), outfield.indexify() + infield.indexify()*3.0)

    # Run the operator
    Operator(stencil, **kwargs)(infield, outfield)

    return outfield


def _new_operator2(shape, time_order, **kwargs):
    infield = TimeData(name='in', shape=shape, time_order=time_order, dtype=np.int32)
    infield.data[:] = np.arange(reduce(mul, shape), dtype=np.int32).reshape(shape)

    outfield = TimeData(name='out', shape=shape, time_order=time_order, dtype=np.int32)

    stencil = Eq(outfield.forward.indexify(),
                 outfield.indexify() + infield.indexify()*3.0)

    # Run the operator
    Operator(stencil, **kwargs)(infield, outfield, t=10)

    return outfield


def test_create_elemental_functions_simple(simple_function):
    old = Rewriter.thresholds['elemental']
    Rewriter.thresholds['elemental'] = 0
    handle = transform(simple_function, mode='split')
    block = List(body=handle.nodes + handle.elemental_functions)
    output = str(block.ccode)
    # Make output compiler independent
    output = [i for i in output.split('\n')
              if all([j not in i for j in ('#pragma', '/*')])]
    assert '\n'.join(output) == \
        ("""void foo(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  float (*restrict c)[5] __attribute__((aligned(64))) = (float (*)[5]) c_vec;
  float (*restrict d)[5][7] __attribute__((aligned(64))) = (float (*)[5][7]) d_vec;
  for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      f_0_0((float*) a,(float*) b,(float*) c,(float*) d,i,j);
    }
  }
}
void f_0_0(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec, const int i, const int j)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  float (*restrict c)[5] __attribute__((aligned(64))) = (float (*)[5]) c_vec;
  float (*restrict d)[5][7] __attribute__((aligned(64))) = (float (*)[5][7]) d_vec;
  for (int k = 0; k < 7; k += 1)
  {
    a[i] = a[i] + b[i] + 5.0F;
    a[i] = -a[i]*c[i][j] + b[i]*d[i][j][k];
  }
}""")
    Rewriter.thresholds['elemental'] = old


def test_create_elemental_functions_complex(complex_function):
    old = Rewriter.thresholds['elemental']
    Rewriter.thresholds['elemental'] = 0
    handle = transform(complex_function, mode='split')
    block = List(body=handle.nodes + handle.elemental_functions)
    output = str(block.ccode)
    # Make output compiler independent
    output = [i for i in output.split('\n')
              if all([j not in i for j in ('#pragma', '/*')])]
    assert '\n'.join(output) == \
        ("""void foo(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  float (*restrict c)[5] __attribute__((aligned(64))) = (float (*)[5]) c_vec;
  float (*restrict d)[5][7] __attribute__((aligned(64))) = (float (*)[5][7]) d_vec;
  for (int i = 0; i < 3; i += 1)
  {
    f_0_0((float*) a,(float*) b,i);
    for (int j = 0; j < 5; j += 1)
    {
      f_0_1((float*) a,(float*) b,(float*) c,(float*) d,i,j);
    }
    f_0_2((float*) a,(float*) b,i);
  }
}
void f_0_0(float *restrict a_vec, float *restrict b_vec, const int i)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  for (int s = 0; s < 4; s += 1)
  {
    b[i] = a[i] + pow(b[i], 2) + 3;
  }
}
void f_0_1(float *restrict a_vec, float *restrict b_vec,"""
         """ float *restrict c_vec, float *restrict d_vec, const int i, const int j)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  float (*restrict c)[5] __attribute__((aligned(64))) = (float (*)[5]) c_vec;
  float (*restrict d)[5][7] __attribute__((aligned(64))) = (float (*)[5][7]) d_vec;
  for (int k = 0; k < 7; k += 1)
  {
    a[i] = a[i]*b[i]*c[i][j]*d[i][j][k];
    a[i] = 4*(a[i] + c[i][j])*(b[i] + d[i][j][k]);
  }
}
void f_0_2(float *restrict a_vec, float *restrict b_vec, const int i)
{
  float (*restrict a) __attribute__((aligned(64))) = (float (*)) a_vec;
  float (*restrict b) __attribute__((aligned(64))) = (float (*)) b_vec;
  for (int q = 0; q < 4; q += 1)
  {
    a[i] = 8.0F*a[i] + 6.0F/b[i];
  }
}""")
    Rewriter.thresholds['elemental'] = old


# Loop blocking tests
# ATM, these tests resemble the ones in test_cache_blocking.py, with the main
# difference being that here the new Operator interface is used

@pytest.mark.parametrize("shape", [(10,), (10, 45), (10, 31, 45)])
@pytest.mark.parametrize("blockshape", [2, 7, (3, 3), (2, 9, 1)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_no_time_loop(shape, blockshape, blockinner):
    wo_blocking = _new_operator1(shape, dle='noop')
    w_blocking = _new_operator1(shape, dle=('blocking', {'blockshape': blockshape,
                                                         'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape", [(20, 33), (45, 31, 45)])
@pytest.mark.parametrize("time_order", [2])
@pytest.mark.parametrize("blockshape", [2, (13, 20), (11, 15, 23)])
@pytest.mark.parametrize("blockinner", [False, True])
def test_cache_blocking_time_loop(shape, time_order, blockshape, blockinner):
    wo_blocking = _new_operator2(shape, time_order, dle='noop')
    w_blocking = _new_operator2(shape, time_order,
                                dle=('blocking', {'blockshape': blockshape,
                                                  'blockinner': blockinner}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


@pytest.mark.parametrize("shape,blockshape", [
    ((25, 25, 46), (None, None, None)),
    ((25, 25, 46), (7, None, None)),
    ((25, 25, 46), (None, None, 7)),
    ((25, 25, 46), (None, 7, None)),
    ((25, 25, 46), (5, None, 7)),
    ((25, 25, 46), (10, 3, None)),
    ((25, 25, 46), (None, 7, 11)),
    ((25, 25, 46), (8, 2, 4)),
    ((25, 25, 46), (2, 4, 8)),
    ((25, 25, 46), (4, 8, 2)),
    ((25, 46), (None, 7)),
    ((25, 46), (7, None))
])
def test_cache_blocking_edge_cases(shape, blockshape):
    wo_blocking = _new_operator2(shape, time_order=2, dle='noop')
    w_blocking = _new_operator2(shape, time_order=2,
                                dle=('blocking', {'blockshape': blockshape,
                                                  'blockinner': True}))

    assert np.equal(wo_blocking.data, w_blocking.data).all()


def test_loop_nofission(simple_function):
    old = Rewriter.thresholds['min_fission'], Rewriter.thresholds['max_fission']
    Rewriter.thresholds['max_fission'], Rewriter.thresholds['min_fission'] = 0, 1
    handle = transform(simple_function, mode='fission')
    assert """\
  for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      for (int k = 0; k < 7; k += 1)
      {
        a[i] = a[i] + b[i] + 5.0F;
        a[i] = -a[i]*c[i][j] + b[i]*d[i][j][k];
      }
    }
  }""" in str(handle.nodes[0].ccode)
    Rewriter.thresholds['min_fission'], Rewriter.thresholds['max_fission'] = old


def test_loop_fission(simple_function_fissionable):
    old = Rewriter.thresholds['min_fission'], Rewriter.thresholds['max_fission']
    Rewriter.thresholds['max_fission'], Rewriter.thresholds['min_fission'] = 0, 1
    handle = transform(simple_function_fissionable, mode='fission')
    assert """\
 for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      for (int k = 0; k < 7; k += 1)
      {
        a[i] = a[i] + b[i] + 5.0F;
      }
      for (int k = 0; k < 7; k += 1)
      {
        b[i] = a[i] + pow(b[i], 2) + 3;
      }
    }
  }""" in str(handle.nodes[0].ccode)
    Rewriter.thresholds['min_fission'], Rewriter.thresholds['max_fission'] = old


def test_padding(simple_function_with_paddable_arrays):
    handle = transform(simple_function_with_paddable_arrays, mode='padding')
    assert str(handle.nodes[0].ccode) == """\
for (int i = 0; i < 3; i += 1)
{
  pa_dense[i] = a_dense[i];
}"""
    assert """\
  for (int i = 0; i < 3; i += 1)
  {
    for (int j = 0; j < 5; j += 1)
    {
      for (int k = 0; k < 7; k += 1)
      {
        pa_dense[i] = b_dense[i] + pa_dense[i] + 5.0F;
      }
    }
  }""" in str(handle.nodes[1].ccode)
    assert str(handle.nodes[2].ccode) == """\
for (int i = 0; i < 3; i += 1)
{
  a_dense[i] = pa_dense[i];
}"""


@pytest.mark.parametrize("shape", [(41,), (20, 33), (45, 31, 45)])
def test_composite_transformation(shape):
    wo_blocking = _new_operator1(shape, dle='noop')
    w_blocking = _new_operator1(shape, dle='advanced')

    assert np.equal(wo_blocking.data, w_blocking.data).all()
