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
from typing import Callable

import numpy as np
import pycuda.autoinit
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

from utils import save_array_as_image, benchmark

MandelFuncT = Callable[[np.ndarray, np.ndarray, int, float], np.ndarray]


def mandelbrot(
    mandel_func: MandelFuncT,
    width: int,
    height: int,
    real_low: float = -2.0,
    real_high: float = 2.0,
    imag_low: float = -2.0,
    imag_high: float = 2.0,
    max_iters: int = 100,
    upper_bound: float = 2.0
):
    real_vals = np.linspace(real_low, real_high, width)
    imag_vals = np.linspace(imag_high, imag_low, height)

    mandelbrot_graph = mandel_func(real_vals, imag_vals, max_iters, upper_bound)

    return mandelbrot_graph


def _build_mandelbrot_kernel():
    return ElementwiseKernel(
        "pycuda::complex<float> *lattice,"
        "float *mandelbrot_graph,"
        "int max_iters,"
        "float upper_bound", """
//cuda
mandelbrot_graph[i] = 1;

pycuda::complex<float> c(lattice[i]);
pycuda::complex<float> z(0, 0);

for (int j = 0; j < max_iters; ++j)
{
    z = z * z + c;

    if (abs(z) > upper_bound)
    {
        mandelbrot_graph[i] = 0;
        break;
    }
}
//!cuda
""", "mandelbrot_kernel"
    )


def mandelbrot_gpu(
    real_vals: np.ndarray, imag_vals: np.ndarray, max_iters: int,
    upper_bound: float
):
    real_vals = real_vals.astype(np.complex64)
    imag_vals = imag_vals.astype(np.complex64) * 1j

    mandelbrot_lattice = real_vals[None, ...] + (imag_vals[..., None])
    mandelbrot_lattice_gpu = gpuarray.to_gpu(mandelbrot_lattice)

    mandelbrot_graph_gpu = gpuarray.empty(
        shape=mandelbrot_lattice.shape, dtype=np.float32
    )

    mandelbrot_ker = _build_mandelbrot_kernel()
    mandelbrot_ker(
        mandelbrot_lattice_gpu, mandelbrot_graph_gpu, np.int32(max_iters),
        np.float32(upper_bound)
    )

    mandelbrot_graph = mandelbrot_graph_gpu.get()

    return mandelbrot_graph


def mandelbrot_cpu(
    real_vals: np.ndarray, imag_vals: np.ndarray, max_iters: int,
    upper_bound: float
):
    mandelbrot_graph = np.ones(
        (len(imag_vals), len(real_vals)), dtype=np.float32
    )

    for x, real_val in enumerate(real_vals):
        for y, imag_val in enumerate(imag_vals):
            c = np.complex64(real_val + imag_val * 1j)
            z = np.complex64(0)

            for _ in range(max_iters):
                z = z**2 + c

                if np.abs(z) > upper_bound:
                    mandelbrot_graph[y, x] = 0
                    break

    return mandelbrot_graph


def main():
    width, height = 512, 512

    for mandel_func, label in (
        (mandelbrot_cpu, "CPU"), (mandelbrot_gpu, "GPU")
    ):
        experiment_name = f"mandelbrot-{label}"

        with benchmark(experiment_name):
            mandelbrot_graph = mandelbrot(mandel_func, width, height)
            save_array_as_image(mandelbrot_graph, f"{experiment_name}.png")

    return 0


if __name__ == '__main__':
    sys.exit(main())
