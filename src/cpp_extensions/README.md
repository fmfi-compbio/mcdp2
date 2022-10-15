to compile the module in-place, run:

```shell
clang++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) cpp_module.cpp -o cpp_module$(python3-config --extension-suffix)
```