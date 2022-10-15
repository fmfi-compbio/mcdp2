import pytest
import os.path

import mcdp2.main
from mcdp2.main import single


def convert_intervals_to_bed(intervals):
    lines = [f"{c}\t{b}\t{e}" for c, b, e in intervals]
    result = "\n".join(lines)
    return result


class TestSimpleBehaviour:
    def test_nonexistent_files_should_throw_ioerror(self, tmp_path):
        reference_filename = ""
        query_filename = ""
        chrlen_filename = ""
        output_dir = tmp_path / "output_dir"

        with pytest.raises(IOError):
            single(reference_filename=reference_filename,
                   query_filename=query_filename,
                   chrlen_filename=chrlen_filename,
                   output_dir=output_dir)

    def test_throw_ioerror_if_output_dir_exists_as_a_file(self, tmp_path):
        reference_filename = tmp_path / "ref.bed"
        reference_filename.write_text(convert_intervals_to_bed([('a', 1, 10), ('a', 20, 30)]))
        query_filename = tmp_path / "query.bed"
        query_filename.write_text(convert_intervals_to_bed([('a', 5, 8), ('a', 10, 12)]))
        chrlen_filename = tmp_path / "genome.txt"
        chrlen_filename.write_text("a\t50\n")
        output_dir = tmp_path / "output_dir"
        output_dir.write_text("blablabla")  # creating the file in place of the intended output dir

        with pytest.raises(IOError):
            single(reference_filename=reference_filename,
                   query_filename=query_filename,
                   chrlen_filename=chrlen_filename,
                   output_dir=output_dir)

    def test_results_should_be_in_output_dir(self, tmp_path):
        reference_filename = tmp_path / "ref.bed"
        reference_filename.write_text(convert_intervals_to_bed([('a', 1, 10), ('a', 20, 30)]))
        query_filename = tmp_path / "query.bed"
        query_filename.write_text(convert_intervals_to_bed([('a', 5, 8), ('a', 10, 12)]))
        chrlen_filename = tmp_path / "genome.txt"
        chrlen_filename.write_text("a\t50\n")
        output_dir = tmp_path / "output_dir"

        single(reference_filename=reference_filename,
               query_filename=query_filename,
               chrlen_filename=chrlen_filename,
               output_dir=output_dir)

        assert os.path.isdir(output_dir)
        assert os.path.isfile(output_dir / mcdp2.main.LOG_FILENAME)
        assert os.path.isfile(output_dir / mcdp2.main.FULL_RESULTS_OUTPUT_FILENAME)

    def test_results_should_be_in_output_dir_even_if_the_dir_already_exists(self, tmp_path):
        reference_filename = tmp_path / "ref.bed"
        reference_filename.write_text(convert_intervals_to_bed([('a', 1, 10), ('a', 20, 30)]))
        query_filename = tmp_path / "query.bed"
        query_filename.write_text(convert_intervals_to_bed([('a', 5, 8), ('a', 10, 12)]))
        chrlen_filename = tmp_path / "genome.txt"
        chrlen_filename.write_text("a\t50\n")
        output_dir = tmp_path / "output_dir"
        os.makedirs(output_dir)

        single(reference_filename=reference_filename,
               query_filename=query_filename,
               chrlen_filename=chrlen_filename,
               output_dir=output_dir)

        assert os.path.isdir(output_dir)
        assert os.path.isfile(output_dir / mcdp2.main.LOG_FILENAME)
        assert os.path.isfile(output_dir / mcdp2.main.FULL_RESULTS_OUTPUT_FILENAME)
