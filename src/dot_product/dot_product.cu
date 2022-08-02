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

struct weighted_mul
{
    weighted_mul(float weight = 1) : weight_(weight) {}

    __device__ float operator()(const float &x, const float &y)
    {
        return weight_ * (x * y);
    }

private:
    float weight_;
};

float dot_product(
    const thrust::device_vector<float> &u,
    const thrust::device_vector<float> &v,
    float weight = 1)
{
    thrust::device_vector<float> elemwise_products(u.size());

    thrust::transform(
        u.begin(),
        u.end(),
        v.begin(),
        elemwise_products.begin(),
        weighted_mul(weight));
    auto weighted_sum(
        thrust::reduce(elemwise_products.begin(), elemwise_products.end()));

    return weighted_sum;
}

int main()
{
    thrust::device_vector<float> vec_1_d;
    thrust::device_vector<float> vec_2_d;

    for (int val = 1; val <= 5; ++val)
    {
        vec_1_d.push_back(static_cast<float>(val));
        vec_2_d.push_back(static_cast<float>(val));
    }

    auto dot_prod_val(dot_product(vec_1_d, vec_2_d));
    cout << "dot-product result: " << dot_prod_val << endl;

    return 0;
}
