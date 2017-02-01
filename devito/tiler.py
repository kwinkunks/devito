from __future__ import absolute_import

import operator
from collections import Iterable, defaultdict
from functools import reduce
from hashlib import sha1
from os import path
from random import randint

import numpy as np
from sympy import Indexed, IndexedBase, symbols
from sympy.utilities.iterables import postorder_traversal

import devito.cgen_wrapper as cgen
from devito.codeprinter import ccode
from devito.compiler import (IntelMICCompiler, get_compiler_from_env,
                             get_tmp_dir, jit_compile_and_load)
from devito.dimension import t, x, y, z
from devito.expression import Expression
from devito.function_manager import FunctionDescriptor, FunctionManager
from devito.iteration import Iteration
from devito.logger import info
from devito.profiler import Profiler
from devito.symbolics import dse_dtype
from devito.tools import flatten


    def generate_tiled_loop(self, loop_body, loop_var, loop_limits, cond_op, loop_step, tile_size):
        tile_var = loop_var+loop_var
        inner_limit = "(%s, %s+%d)" % (loop_limits[1], tile_var, tile_size)
        # Is self._forward generic enough? Should cond_op be used to reason about
        # iteration direction?
        limit_min_max = "MIN" if self._forward else "MAX"

        # Create inner loop with limits [tile_var, min/max(loop_limit, tile_var + tile_size))
        inner = cgen.For(
            cgen.InlineInitializer(cgen.Value("int", loop_var), tile_var),
            loop_var + cond_op + limit_min_max + inner_limit,
            loop_var + "+=" + str(loop_step),
            loop_body
        )

        # Create outer loop bounded by loop_limit, with tile_size sized steps
        loop_body = cgen.For(
            cgen.InlineInitializer(cgen.Value("int", tile_var), str(loop_limits[0])),
            tile_var + cond_op + str(loop_limits[1]),
            tile_var + "+=" + str(tile_size),
            inner
        )
        return loop_body


