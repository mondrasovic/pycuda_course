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

import click
import numpy as np
import pycuda.autoinit
from matplotlib import animation
from matplotlib import pyplot as plt
from pycuda import gpuarray
from pycuda.compiler import SourceModule


def build_life_simulation_kernel() -> SourceModule:
    return SourceModule(
        """
//cuda
#define X (threadIdx.x + (blockIdx.x * blockDim.x))
#define Y (threadIdx.y + (blockIdx.y * blockDim.y))

#define WIDTH (blockDim.x * gridDim.x)
#define HEIGHT (blockDim.y * gridDim.y)

#define XM(x) ((x + WIDTH) % WIDTH)
#define YM(y) ((y + HEIGHT) % HEIGHT)

#define INDEX(x, y) (XM(x) + (YM(y) * WIDTH))

#define IS_ALIVE(lattice, index) ((lattice)[index] == 1)

__device__ int count_neighbors(int x, int y, int *in)
{
    return (
        in[INDEX(x - 1, y - 1)] +
        in[INDEX(x, y - 1)] +
        in[INDEX(x + 1, y - 1)] +
        in[INDEX(x - 1, y)] +
        in[INDEX(x + 1, y)] +
        in[INDEX(x - 1, y + 1)] +
        in[INDEX(x + 1, y)] +
        in[INDEX(x + 1, y + 1)]
    );
}

__global__ void life_simul_step_ker(int *lattice_in, int *lattice_out)
{
    const int x = X;
    const int y = Y;
    const int index = INDEX(x, y);

    int n_neighbors = count_neighbors(x, y, lattice_in);

    if (IS_ALIVE(lattice_in, index))
    {
        switch (n_neighbors)
        {
            case 2:
            case 3:
                lattice_out[index] = 1;
                break;
            default:
                lattice_out[index] = 0;
        }
    }
    else
    {
        lattice_out[index] = (n_neighbors == 3) ? 1 : 0;
    }
}
//!cuda
"""
    )


@click.command()
@click.option(
    '-s',
    '--lattice-size',
    type=int,
    default=512,
    show_default=True,
    help="Lattice side size."
)
def main(lattice_size: int) -> int:
    life_simul_ker = build_life_simulation_kernel()
    life_simul_step_ker = life_simul_ker.get_function('life_simul_step_ker')

    def update_gpu(frame_num, image, curr_lattice_gpu, new_lattice_gpu):
        life_simul_step_ker(
            curr_lattice_gpu,
            new_lattice_gpu,
            grid=(lattice_size // 32, lattice_size // 32, 1),
            block=(32, 32, 1)
        )

        image.set_data(new_lattice_gpu.get())
        curr_lattice_gpu[:] = new_lattice_gpu[:]

        return image

    curr_lattice = np.random.choice(
        [0, 1], size=(lattice_size, lattice_size), p=[0.75, 0.25]
    )
    curr_lattice_gpu = gpuarray.to_gpu(curr_lattice)
    new_lattice_gpu = gpuarray.empty_like(curr_lattice_gpu)

    fig, ax = plt.subplots(figsize=(10, 10))
    image = ax.imshow(curr_lattice_gpu.get(), interpolation='nearest')
    anim = animation.FuncAnimation(
        fig,
        update_gpu,
        fargs=(image, curr_lattice_gpu, new_lattice_gpu),
        interval=100,
        frames=1000,
        save_count=1000
    )
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
