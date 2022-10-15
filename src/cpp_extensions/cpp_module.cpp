#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

int add(int i, int j) {
    return i + j;
}

// // Python implementation:
//def compute_joint_pmf(pmfs):
//    if len(pmfs) == 0:
//        return []
//    elif len(pmfs) == 1:
//        return pmfs[0]
//
//    current_line = pmfs[0]
//    for num in range(1, len(pmfs)):
//        pmf = pmfs[num]
//        new_total_length = len(current_line) + len(pmf) - 1
//        next_line = [0 for _ in range(new_total_length)]
//        for i in range(new_total_length):
//            # 0 <= j < len(current_line) && 0 <= i - j < len(pmf)
//            # --> 0 >= j - i > - len(pmf)
//            # --> i >= j > i - len(pmf)
//            # -----> max(0, i - len(pmf)+1) <= j < min(i+1, len(current_line))
//            for j in range(max(0, i - len(pmf) + 1), min(i + 1, len(current_line))):
//                next_line[i] += current_line[j] * pmf[i - j]
//        current_line = next_line
//    return current_line

py::array_t<long double> compute_joint_pmf_flattened(
    py::array_t<long double> flat_data,
    py::array_t<int> sizes
) {
    auto flat_data_request = flat_data.request();
    auto sizes_request = sizes.request();
    const int count = sizes_request.shape[0];
    auto flat_data_ptr = static_cast<long double *>(flat_data_request.ptr);
    auto sizes_ptr = static_cast<int *>(sizes_request.ptr);

    if (count <= 1) {
        return flat_data;
    }

    std::vector<long double> current_line(flat_data_ptr, flat_data_ptr+sizes_ptr[0]);
    current_line.reserve(flat_data_request.shape[0]-count+1);
    std::vector<long double> next_line;
    next_line.reserve(flat_data_request.shape[0]-count+1);
    auto pmf_start = flat_data_ptr+sizes_ptr[0];
    for (auto num = 1; num < count; num++) {
        const auto new_total_length = current_line.size() + sizes_ptr[num] - 1;
        next_line.clear();
        next_line.resize(new_total_length, 0.0);
//        std::vector<long double> next_line(new_total_length, 0.0);
        for (int i = 0; i < new_total_length; i++) {
            for (int j = std::max(0, i - sizes_ptr[num] + 1); j < std::min(i+1, static_cast<int>(current_line.size())); j++) {
                next_line[i] += current_line[j] * pmf_start[i - j];
            }
        }
        swap(current_line, next_line);
        pmf_start += sizes_ptr[num];
    }

    long double* result = new long double[static_cast<int>(current_line.size())];
    for (int i = 0; i < current_line.size(); i++) {
        result[i] = current_line[i];
    }

    return py::array_t<long double>(static_cast<int>(current_line.size()), result);
}

PYBIND11_MODULE(cpp_module, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
    m.def("compute_joint_pmf_flattened",
        &compute_joint_pmf_flattened,
        "Compute joint PMF for flattened PMFs.");
}