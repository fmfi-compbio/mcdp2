import cpp_extensions.cpp_module


class TestSimpleAddition:
    def test_call(self):
        i, j = 1, 2
        expected = i + j

        result = cpp_extensions.cpp_module.add(i, j)
        assert result == expected
