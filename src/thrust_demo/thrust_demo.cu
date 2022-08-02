// Copyright (c) 2022 Milan Ondrasovic, milan.ondrasovic@gmail.com
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files(the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and / or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT.IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
// LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
// OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include <iostream>
#include <string>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

using namespace std;

int main()
{
    thrust::host_vector<int> vec_h;

    vec_h.push_back(0);
    vec_h.push_back(1);
    vec_h.push_back(2);

    thrust::device_vector<int> vec_d(vec_h); // Simple transfer to GPU!

    vec_d.push_back(3);

    string sep("");
    for (const auto &val : vec_d)
    {
        cout << sep << val;
        sep = ", ";
    }
    cout << endl;

    return 0;
}
