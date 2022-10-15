import io
import pytest

from mcdp2.common.io import load_chromosome_lengths, \
    load_intervals_from_bed, _parse_bed_intervals, \
    _is_bed_header_line, create_default_context, load_context, _make_context_intervals_from_records, \
    create_context_from_masking


class TestLoadChromosomeLengths:
    def test_empty_file(self):
        fd = io.StringIO("")
        expected = []

        result = load_chromosome_lengths(fd)
        assert result == expected

    @pytest.mark.parametrize("text,expected", [
        ("a\t10\n", [('a', 10)]),
        ("a\t10\n \nb\t20", [('a', 10), ('b', 20)]),
    ])
    def test_some_lines(self, text, expected):
        fd = io.StringIO(text)
        result = load_chromosome_lengths(fd)
        assert result == expected

    @pytest.mark.parametrize("text", [
        "chr1\t10\nchr2 20\n",
        "chr1\t10\nchr2\t20\tgarbage\n"
    ])
    def test_should_fail_for_not_two_columns(self, text):
        fd = io.StringIO(text)
        with pytest.raises(ValueError):
            result = load_chromosome_lengths(fd)

    @pytest.mark.parametrize('text', [
        "c\t10\nd\t10\nc\t20\n"
    ])
    def test_should_fail_for_duplicate_names(self, text):
        fd = io.StringIO(text)
        with pytest.raises(ValueError):
            result = load_chromosome_lengths(fd)


class TestParseBedIntervals:
    def test_empty_file(self):
        fd = ""
        expected = ([], [])

        result = _parse_bed_intervals(io.StringIO(fd))
        assert result == expected

    cases_headers = [
        ('browser blabla\n', (['browser blabla'], [])),
        ('trackblabla', (['trackblabla'], [])),
        ('browser bla\n# hehe\n', (['browser bla', "# hehe"], [])),
    ]

    cases_records = [
        ('t1\t10\t20\n', ([], [('t1', 10, 20)])),
        ('t1\t10\t20\tname', ([], [('t1', 10, 20, "name")])),
        ('t1\t10\t20\tname\t470', ([], [('t1', 10, 20, "name", 470)])),
        ('t1\t10\t20\tname\t470\t+', ([], [('t1', 10, 20, "name", 470, "+")])),
        ('t1\t10\t20\tname\t470\t+\t100\t120', ([], [('t1', 10, 20, "name", 470, "+", 100, 120)])),
        ('t1\t10\t20\tname\t470\t+\t100\t120\t0,1,2',
         ([], [('t1', 10, 20, "name", 470, "+", 100, 120, [0, 1, 2])])),
        ('t1\t10\t20\tname\t470\t+\t100\t120\t0,1,2\t2\t3,4\t0,5',
         ([], [('t1', 10, 20, "name", 470, "+", 100, 120, [0, 1, 2], 2, [3, 4], [0, 5])])),
    ]

    @pytest.mark.parametrize("fd,expected", cases_headers + cases_records)
    def test_parse_files(self, fd, expected):
        result = _parse_bed_intervals(io.StringIO(fd))
        assert result == expected

    cases_incorrect_number_of_columns = [
        '12\n',
        '1\t2\n',
        '\t'.join(str(i) for i in range(13)),
        '\t'.join(str(i) for i in range(20)),
    ]

    @pytest.mark.parametrize("fd", cases_incorrect_number_of_columns)
    def test_incorrect_number_of_columns(self, fd):
        with pytest.raises(ValueError):
            result = _parse_bed_intervals(io.StringIO(fd))


@pytest.mark.parametrize("line,expected", [
    ("", False),
    ("\n", False),
    ("browser\n", True),
    ("browser", True),
    ("browserdfsdflsdjfkl jsdklfjkdlsjlkdj\n", True),
    ("#fsfsefsefsef\n", True),
    ("tracksfsefefse", True),
])
def test_is_bed_header_line(line, expected):
    result = _is_bed_header_line(line)
    assert result == expected


class TestLoadIntervalsFromBed:
    @pytest.mark.parametrize("text,expected", [
        ("", []),
        ("#fsefesfse", []),
        ("name\t10\t20\nname2\t11\t12\tblabla\n", [('name', 10, 20), ('name2', 11, 12)]),
        ('t1\t10\t20\tname\t470\t+\t100\t120\t0,1,2\t2\t3,4\t0,5',
         [('t1', 10, 20)]),
    ])
    def test_ignore_blocks(self, text, expected):
        fd = io.StringIO(text)
        result = load_intervals_from_bed(fd, ignore_blocks=True)
        assert result == expected

    @pytest.mark.parametrize("text,expected", [
        ("", []),
        ("#fsefesfse", []),
        ("name\t10\t20\nname2\t11\t12\tblabla\n", [('name', 10, 20), ('name2', 11, 12)]),
        ('t1\t10\t20\tname\t470\t+\t100\t120\t0,1,2\t2\t3,4\t0,5',
         [('t1', 10, 13), ('t1', 15, 19)]),
    ])
    def test_use_blocks(self, text, expected):
        fd = io.StringIO(text)
        result = load_intervals_from_bed(fd, ignore_blocks=False)
        assert result == expected


@pytest.mark.parametrize("chromosome_names,expected", [
    ([('a', 2)], {'a': [('a', 0, 2)]}),
    ([('a', 3)], {'a': [('a', 0, 3)]}),
    ([('a', 3), ('b', 10), ('c', 20)],
     {'a': [('a', 0, 3)], 'b': [('b', 0, 10)], 'c': [('c', 0, 20)]})
])
def test_create_default_context(chromosome_names, expected):
    result = create_default_context(chromosome_names, separate_models=True)
    assert result == expected


class TestLoadContext:
    # def test_empty_file_returns_error(self):
    #     fd = io.StringIO("")
    #     with pytest.raises(ValueError):
    #         result = load_context(fd, None, ignore_blocks=False)

    @pytest.mark.parametrize("text,chromosome_lengths,expected", [
        ("c\t0\t20\tA\nc\t20\t40\tB\n", [("c", 40)], {'c': [('A', 0, 20), ('B', 20, 40)]}),
    ])
    def test_working_examples(self, text, chromosome_lengths, expected):
        fd = io.StringIO(text)
        result = load_context(fd, chromosome_lengths, ignore_blocks=False)
        assert result == expected

    @pytest.mark.parametrize("text,chromosome_lengths", [
        ("c\t10\t20\tA\nc\t15\t25\tB\n", [("c", 50)])
    ])
    def test_fail_for_overlapping_intervals(self, text, chromosome_lengths):
        fd = io.StringIO(text)
        with pytest.raises(ValueError):
            result = load_context(fd, chromosome_lengths, ignore_blocks=False)


@pytest.mark.parametrize("records,ignore_blocks,expected", [
    ([('c', 10, 20, 'A')], False, {'c': [('A', 10, 20)]}),
    ([('c', 10, 20, 'A'), ('c', 0, 100, 'B', None, None, None, None, None, 3, [10, 20, 30], [0, 15, 40])],
     True, {'c': [('A', 10, 20), ('B', 0, 100)]}),
    ([('c', 10, 20, 'A'), ('c', 0, 100, 'B', None, None, None, None, None, 3, [10, 20, 30], [0, 15, 40])],
     False, {'c': [('A', 10, 20), ('B', 0, 10), ('B', 15, 35), ('B', 40, 70)]}),
])
def test_make_context_intervals_from_records(records, ignore_blocks, expected):
    result = _make_context_intervals_from_records(records, ignore_blocks)
    assert result == expected


@pytest.mark.parametrize("intervals,chromosome_lengths,expected", [
    ([], {'c1': 10}, {'c1': [('c1_unmasked', 0, 10)]}),
    ([], {'c1': 10, 'c2': 20},
     {'c1': [('c1_unmasked', 0, 10)], 'c2': [('c2_unmasked', 0, 20)]}),
    ([('c1', 3, 8)], {'c1': 10, 'c2': 20},
     {'c1': [('c1_unmasked', 0, 3), ('c1_masked', 3, 8), ('c1_unmasked', 8, 10)], 'c2': [('c2_unmasked', 0, 20)]}),
])
def test_create_context_from_masking(intervals, chromosome_lengths, expected):
    result = create_context_from_masking(intervals, chromosome_lengths, separate_models=True)
    assert result == expected
