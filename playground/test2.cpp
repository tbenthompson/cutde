// cppimport
#include <pybind11/pybind11.h>

namespace py = pybind11;

int square(int x) {
    return x * x;
}

PYBIND11_MODULE(test2, m) {
    m.def("square", &square);
}
/*
<%
setup_pybind11(cfg)
%>
*/
