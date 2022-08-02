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
