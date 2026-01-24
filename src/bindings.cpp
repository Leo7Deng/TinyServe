#include <pybind11/pybind11.h>
// #include <cuda_runtime.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

PYBIND11_MODULE(tinyserve_ext, m, py::mod_gil_not_used()) {
    m.def("add", &add, "A function that adds two numbers");
}