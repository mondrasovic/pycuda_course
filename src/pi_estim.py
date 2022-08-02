#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (c) 2022 Milan Ondrasovic, milan.ondrasovic@gmail.com
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
# LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
# WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import sys

import numpy as np
import pycuda.autoinit
import sympy as sp
from pycuda import driver
from pycuda import gpuarray
from pycuda.compiler import SourceModule


def build_pi_estim_monte_carlo_kernel():
    return SourceModule(
        """
//cuda
#include <curand_kernel.h>

typedef unsigned long long int ull_t;

extern "C"
{
__global__ void estimate_pi(ull_t *n_hits, ull_t n_iters)
{
    curandState cr_state;
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    curand_init(
        (ull_t) clock() + (ull_t) tid, 0, 0, &cr_state);
    
    while (n_iters-- > 0)
    {
        float x = curand_uniform(&cr_state);
        float y = curand_uniform(&cr_state);

        if ((x * x) + (y * y) <= 1.0f)
        {
            ++n_hits[tid];
        }
    }
}
}
//!cuda
""",
        no_extern_c=True
    )


def main() -> int:
    estimate_pi_ker = build_pi_estim_monte_carlo_kernel()
    estimate_pi = estimate_pi_ker.get_function('estimate_pi')

    n_threads_per_block = 32
    n_blocks_per_grid = 512
    n_total_threads = n_threads_per_block * n_blocks_per_grid

    n_hits_results_d = gpuarray.zeros((n_total_threads, ), dtype=np.uint64)

    n_iters = 2**24

    estimate_pi(
        n_hits_results_d,
        np.uint64(n_iters),
        grid=(n_blocks_per_grid, 1, 1),
        block=(n_threads_per_block, 1, 1)
    )

    n_hits_results_h = n_hits_results_d.get()
    n_total_hits = np.sum(n_hits_results_h)
    n_total_samples = n_total_threads * n_iters

    estim_pi_symbolic = (
        sp.Rational(4) * sp.Rational(n_total_hits, n_total_samples)
    )
    estim_pi_val = estim_pi_symbolic.evalf()

    print(f"Estimated PI value: {estim_pi_val:0.12f}")

    return 0


if __name__ == '__main__':
    sys.exit(main())