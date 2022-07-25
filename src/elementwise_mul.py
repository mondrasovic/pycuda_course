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
from pycuda import gpuarray
from pycuda.elementwise import ElementwiseKernel

from utils import benchmark

N_VALS = 50_000_000


def build_elementwise_2x_kernel():
    return ElementwiseKernel(
        "float *in, float *out", "out[i] = 2 * in[i];", "gpu_2x_ker"
    )


def main():
    host_data = np.float32(np.random.random(N_VALS))

    with benchmark("CPU"):
        host_data_2x = host_data * 2

    gpu_2x_ker = build_elementwise_2x_kernel()
    device_data = gpuarray.to_gpu(host_data)
    device_data_2x = gpuarray.empty_like(device_data)

    with benchmark("GPU"):
        gpu_2x_ker(device_data, device_data_2x)

    from_device_data_2x = device_data_2x.get()

    assert np.allclose(host_data_2x, from_device_data_2x)

    return 0


if __name__ == '__main__':
    sys.exit(main())
