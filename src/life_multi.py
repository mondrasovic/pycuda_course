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
from typing import Callable, Sequence

import click
import numpy as np
import pycuda.autoinit
from matplotlib import animation
from matplotlib import pyplot as plt
from pycuda import driver
from pycuda import gpuarray
from pycuda.compiler import SourceModule


class Simulation:
    def __init__(
        self,
        life_simul_step_ker_exec: Callable,
        lattice_size: int,
        cell_occur_prob: float = 0.25
    ) -> None:
        self.life_simul_step_ker_exec = life_simul_step_ker_exec

        self._stream = driver.Stream()

        curr_lattice = np.random.choice(
            [0, 1],
            size=(lattice_size, lattice_size),
            p=[(1 - cell_occur_prob), cell_occur_prob]
        )
        self._curr_lattice_gpu = gpuarray.to_gpu(curr_lattice)
        self._new_lattice_gpu = gpuarray.empty_like(self._curr_lattice_gpu)

    def step(self) -> np.ndarray:
        self.life_simul_step_ker_exec(
            self._curr_lattice_gpu, self._new_lattice_gpu
        )

        self._curr_lattice_gpu.set_async(
            self._new_lattice_gpu, stream=self._stream
        )

        return self.get_current_lattice_image_data()

    def get_current_lattice_image_data(self) -> np.ndarray:
        return self._curr_lattice_gpu.get_async(stream=self._stream)


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
@click.option(
    '-n',
    '--concurrent-num',
    type=int,
    default=4,
    show_default=True,
    help="Number of concurrent simulations."
)
@click.option(
    '-p',
    '--cell-occur-prob',
    type=float,
    default=0.25,
    show_default=True,
    help="Probability of a cell occupying a given position at initialization."
)
def main(lattice_size: int, concurrent_num: int, cell_occur_prob: float) -> int:
    life_simul_ker = build_life_simulation_kernel()
    life_simul_step_ker = life_simul_ker.get_function('life_simul_step_ker')

    def life_simul_step_ker_exec(curr_lattice_gpu, new_lattice_gpu) -> None:
        life_simul_step_ker(
            curr_lattice_gpu,
            new_lattice_gpu,
            grid=(lattice_size // 32, lattice_size // 32, 1),
            block=(32, 32, 1)
        )

    def update_gpu(frame_num, images, simulations):
        for image, simulation in zip(images, simulations):
            image_data = simulation.step()
            image.set_data(image_data)

        return images

    simulations = [
        Simulation(life_simul_step_ker_exec, lattice_size, cell_occur_prob)
        for _ in range(concurrent_num)
    ]

    fig, axes = plt.subplots(nrows=1, ncols=concurrent_num, figsize=(10, 10))

    images = []
    for ax, simulation in zip(axes, simulations):
        image = ax.imshow(
            simulation.get_current_lattice_image_data(),
            interpolation='nearest'
        )
        images.append(image)

    anim = animation.FuncAnimation(
        fig,
        update_gpu,
        fargs=(images, simulations),
        interval=100,
        frames=1000,
        save_count=1000
    )

    fig.tight_layout()
    plt.show()

    return 0


if __name__ == '__main__':
    sys.exit(main())
