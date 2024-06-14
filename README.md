# MCDP2 - colocalization analysis for genomic annotations with genomic contexts
## A tool to compute the significance of enrichment (depletion) between two genomic annotations using number of overlaps or number of shared bases as the test statistic

## Install 

```shell
conda create -n mcdp2 python=3.10 
conda activate mcdp2
conda install pybind11
python -m pip install .
mcdp2 --help
```

OR, using `conda`:

```shell
conda install japdlsd::mcdp2
```

## Demo

```shell
conda activate mcdp2
cd demo
bash run_comparison.sh
```

The program produces three files in the output directory:

1. `summary.txt` - Results (z-score, p-value) and general statistics about the input data in the plain-text version
2. `data.yaml` - Structured dump of all results, including full PMF function for `--exact` flag and subresults for the individual chromosomes
3. `log` - log file for debug purposes

## Quick usage

### Test of a single reference annotation enrichment

```shell
# test on a single-class context
mcdp2 single ref.bed query.bed chrom.sizes -o output_directory

# test with a masking
mcdp2 single ref.bed query.bed chrom.sizes -m masking.bed -o output_directory

# test with context
mcdp2 single ref.bed query.bed chrom.sizes -c context.bed -o output_directory
```

### Test of relative enrichment for two disjoint reference annotations

```shell
mcdp2 diff-disjoint ref1.bed ref2.bed query.bed chrom.sizes -o output_directory
```

### Test of relative enrichment for a subset reference annotation

```shell
mcdp2 diff-subset ref_subset.bed ref_superset.bed query.bed chrom.sizes -o output_directory
```

## Full usage

```
 mcdp2 single [-h] [-c CONTEXT_FILENAME] [-m MASKING_FILENAME] [-o OUTPUT_DIR]
                    [--ignore-bed-blocks] [-t THREADS] [-s {overlaps,bases}]
                    [-a {hybrid,exact,fast,both}]
                    [--hybrid-threshold HYBRID_THRESHOLD] [--separate-models] [-d]
                    reference-filename query-filename chrlen-filename

Test if the reference has significantly many overlaps/shared bases with the query. Allows to use masking or context. The
    references have to be disjoint. 

positional arguments:
  reference-filename    BED file with the reference annotation
  query-filename        BED file with the query annotation
  chrlen-filename       TXT file with chromosome lengths

options:
  -h, --help            show this help message and exit
  -c CONTEXT_FILENAME, --context-filename CONTEXT_FILENAME
                        BED file with context for query annotation. Fourth column
                        would be used as identifier. Intervals with different
                        identifiers must be non-overlapping. If not specified, the
                        individual chromosomes would be used. (default: -)
  -m MASKING_FILENAME, --masking-filename MASKING_FILENAME
                        BED file with intervals to mask out. This is achieved by
                        removing the masking intervals from the input annotations and
                        creating a separate context for the masking intervals. Since
                        this feature will produce its own context, it cannot be used
                        together with the '-c' flag. (default: -)
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory (default: 'mcdp2-results-20221015T164745')
  --ignore-bed-blocks   If set, the intervals are created using chromStart and
                        chromEnd, ingoring the blocks in BED files (including context
                        and masking intervals!) (default: False)
  -t THREADS, --threads THREADS
                        Number of threads to use. (default: 4)
  -s {overlaps,bases}, --statistics {overlaps,bases}
                        The test statistics to be used in the tests. 'overlaps' means
                        the number of intervals in reference annotation hit by an
                        interval in query. 'bases' means the number of shared bases
                        between the reference and query annotations. (default:
                        'overlaps')
  -a {hybrid,exact,fast,both}, --algorithm {hybrid,exact,fast,both}
                        Algorithm to compute the p-value (only for '-s overlaps'
                        option). 'fast' computes a normal approximationof the total
                        PMF and is the faster option. 'exact' computes the PMF using a
                        quadratic DP algorithm and is slower, but it is better for
                        smaller datasets. 'hybrid' selects the algorithm based on the
                        total size of the reference. Once the total size is greater
                        than 'hybrid_threshold', it switches to the fast version.
                        'both' will run both algorithms. The option is ignored if the
                        number of shared bases is chosen as the test statistic.
                        (default: 'hybrid')
  --hybrid-threshold HYBRID_THRESHOLD
                        A threshold to switch from 'exact' to 'fast' algorithm for
                        'hybrid' option. (default: 50000)
  --separate-models     If set, the individual chromosomes have separate models.
                        Otherwise the underlying model is the same. Ignored is the
                        context is provided. (default: False)
  -d, --logging-debug   If enabled, the logging level is set to `logging.DEBUG`
                        (default: False)
``` 

```
mcdp2 diff-subset [-h] [-o OUTPUT_DIR] [--ignore-bed-blocks] [-t THREADS]
                         [-s {overlaps,bases}] [-a {hybrid,exact,fast,both}]
                         [--hybrid-threshold HYBRID_THRESHOLD] [-d]
                         [--separate-models]
                         subset-filename superset-filename query-filename
                         chrlen-filename

Test if the subset reference has significantly more overlaps/shared bases than the superset reference.

positional arguments:
  subset-filename       BED file with the subset reference annotation
  superset-filename     BED file with the superset reference annotation
  query-filename        BED file with the query annotation
  chrlen-filename       TXT file with chromosome lengths

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory (default: 'mcdp2-results-20221015T164517')
  --ignore-bed-blocks   If set, the intervals are created using chromStart and
                        chromEnd (default: False)
  -t THREADS, --threads THREADS
                        Number of threads to use. (default: 4)
  -s {overlaps,bases}, --statistics {overlaps,bases}
                        The test statistics to be used in the tests. 'overlaps' means
                        the number of intervals in reference annotation hit by an
                        interval in query. 'bases' means the number of shared bases
                        between the reference and query annotations. (default:
                        'overlaps')
  -a {hybrid,exact,fast,both}, --algorithm {hybrid,exact,fast,both}
                        Algorithm to compute the p-value (only for '-s overlaps'
                        option). 'fast' computes a normal approximationof the total
                        PMF and is the faster option. 'exact' computes the PMF using a
                        quadratic DP algorithm and is slower, but it is better for
                        smaller datasets. 'hybrid' selects the algorithm based on the
                        total size of the reference. Once the total size is greater
                        than 'hybrid_threshold', it switches to the fast version.
                        'both' will run both algorithms. The option is ignored if the
                        number of shared bases is chosen as the test statistic.
                        (default: 'hybrid')
  --hybrid-threshold HYBRID_THRESHOLD
                        A threshold to switch from 'exact' to 'fast' algorithm for
                        'hybrid' option. (default: 50000)
  -d, --logging-debug   If enabled, the logging level is set to `logging.DEBUG`
                        (default: False)
  --separate-models     If set, the individual chromosomes have separate models.
                        Otherwise the underlying model is the same. (default: False)
```

```
mcdp2 diff-disjoint [-h] [-o OUTPUT_DIR] [--ignore-bed-blocks] [-t THREADS]
                           [-a {hybrid,exact,fast,both}] [-s {overlaps,bases}]
                           [--hybrid-threshold HYBRID_THRESHOLD] [-d]
                           [--separate-models]
                           ref1-filename ref2-filename query-filename chrlen-filename

Test if the first reference has significantly more overlaps/shared bases than the second one.

positional arguments:
  ref1-filename         BED file with the first reference annotation
  ref2-filename         BED file with the second reference annotation
  query-filename        BED file with the query annotation
  chrlen-filename       TXT file with chromosome lengths

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory (default: 'mcdp2-results-20221015T164539')
  --ignore-bed-blocks   If set, the intervals are created using chromStart and
                        chromEnd (default: False)
  -t THREADS, --threads THREADS
                        Number of threads to use. (default: 4)
  -a {hybrid,exact,fast,both}, --algorithm {hybrid,exact,fast,both}
                        Algorithm to compute the p-value (only for '-s overlaps'
                        option). 'fast' computes a normal approximationof the total
                        PMF and is the faster option. 'exact' computes the PMF using a
                        quadratic DP algorithm and is slower, but it is better for
                        smaller datasets. 'hybrid' selects the algorithm based on the
                        total size of the reference. Once the total size is greater
                        than 'hybrid_threshold', it switches to the fast version.
                        'both' will run both algorithms. The option is ignored if the
                        number of shared bases is chosen as the test statistic.
                        (default: 'hybrid')
  -s {overlaps,bases}, --statistics {overlaps,bases}
                        The test statistics to be used in the tests. 'overlaps' means
                        the number of intervals in reference annotation hit by an
                        interval in query. 'bases' means the number of shared bases
                        between the reference and query annotations. (default:
                        'overlaps')
  --hybrid-threshold HYBRID_THRESHOLD
                        A threshold to switch from 'exact' to 'fast' algorithm for
                        'hybrid' option. (default: 50000)
  -d, --logging-debug   If enabled, the logging level is set to `logging.DEBUG`
                        (default: False)
  --separate-models     If set, the individual chromosomes have separate models.
                        Otherwise the underlying model is the same. (default: False)
```


## Acknowledgements

Please cite this tool as follows:

> Gafurov, A., Vinař, T., Medvedev, P., Brejová, B. (2024). Efficient Analysis of Annotation Colocalization Accounting for Genomic Contexts. In: Ma, J. (eds) Research in Computational Molecular Biology. RECOMB 2024. Lecture Notes in Computer Science, vol 14758. Springer, Cham. https://doi.org/10.1007/978-1-0716-3989-4_3



[//]: # (## For development)

[//]: # ()
[//]: # (### Install locally)

[//]: # ()
[//]: # (```shell)

[//]: # (python -m pip install . )

[//]: # (```)

[//]: # ()
[//]: # (### Build conda package & install it)

[//]: # ()
[//]: # (```shell)

[//]: # (conda build purge)

[//]: # (conda build -c conda-forge -c bioconda .)

[//]: # (conda install --use-local --force-reinstall mcdp2 -y)

[//]: # (```)

[//]: # ()
[//]: # (### Work build with pybind11 stuff)

[//]: # ()
[//]: # (```shell)

[//]: # (conda install -c conda-forge argh pybind11)

[//]: # (cd src/cpp_extensions/)

[//]: # (c++ -O3 -Wall -shared -std=c++11 -fPIC $&#40;python3 -m pybind11 --includes&#41; cpp_module.cpp -o cpp_module$&#40;python3-config --extension-suffix&#41;)

[//]: # (cd -)

[//]: # (```)

