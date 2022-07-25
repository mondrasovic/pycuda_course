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

import time

from matplotlib import pyplot as plt
from matplotlib import cm


def save_array_as_image(array, file_name):
    plt.imsave(file_name, array, cmap=cm.gray)


class benchmark:
    def __init__(self, name):
        self.name = name
        self._start_time = None

    def __enter__(self):
        self._start_time = time.time()

    def __exit__(self, ty, val, tb):
        end_time = time.time()
        time_diff = end_time - self._start_time
        print(f"Elapsed time for '{self.name}': {time_diff:.8f} second(s).")
