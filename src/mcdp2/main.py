import os.path
import yaml
import argh
import logging
from datetime import datetime

from mcdp2.common.io import load_intervals_from_bed, load_chromosome_lengths, load_context, create_default_context, \
    create_context_from_superset, create_context_from_masking
from mcdp2.common.helpers import filter_intervals_by_chrnames, merge_nondisjoint_intervals, \
    mask_intervals
from mcdp2.models.overlaps import compute_pvalue as compute_pvalue_overlaps
from mcdp2.models.shared_bases import compute_pvalue as compute_pvalue_bases

FULL_RESULTS_OUTPUT_FILENAME = "data.yaml"
SUMMARY_OUTPUT_FILENAME = "summary.txt"
LOG_FILENAME = "log"


@argh.arg("reference-filename",
          help="BED file with the reference annotation")
@argh.arg("query-filename",
          help="BED file with the query annotation")
@argh.arg("chrlen-filename",
          help="TXT file with chromosome lengths")
@argh.arg("-c", "--context-filename",
          help="BED file with context for query annotation. Fourth column would be used as "
               "identifier. Intervals with different identifiers must be non-overlapping. If not "
               "specified, the individual chromosomes would be used.")
@argh.arg("-m", "--masking-filename",
          help="BED file with intervals to mask out. This is achieved by removing the masking "
               "intervals from the input annotations and creating a separate context for the "
               "masking intervals. Since this feature "
               "will produce its own context, it cannot be used together with the '-c' flag.")
@argh.arg("-o", "--output-dir",
          help="Output directory")
@argh.arg("--ignore-bed-blocks",
          help="If set, the intervals are created using chromStart and chromEnd, ingoring the "
               "blocks in BED files (including context and masking intervals!)")
@argh.arg("-t", "--threads",
          help="Number of threads to use.")
@argh.arg("-s", "--statistics", choices=["overlaps", "bases"],
          help="The test statistics to be used in the tests. 'overlaps' means the number of intervals in reference "
               "annotation hit by an interval in query. 'bases' means the number of shared bases between the "
               "reference and query annotations.")
@argh.arg("-a", "--algorithm", choices=["hybrid", "exact", "fast", 'both'],
          help="Algorithm to compute the p-value (only for '-s overlaps' option). "
               "'fast' computes a normal approximation"
               "of the total PMF and is the faster option. 'exact' computes the PMF using "
               "a quadratic DP algorithm and is slower, but it is better for smaller datasets. "
               "'hybrid' selects the algorithm based on the total size of the reference. "
               "Once the total size is greater than 'hybrid_threshold', "
               "it switches to the fast version. "
               "'both' will run both algorithms. "
               "The option is ignored if the number of shared bases is chosen as the test statistic.")
@argh.arg("--hybrid-threshold",
          help="A threshold to switch from 'exact' to 'fast' algorithm for 'hybrid' option.")
@argh.arg("--separate-models", help="If set, the individual chromosomes have separate models. "
                                    "Otherwise the underlying model is the same. "
                                    "Ignored is the context is provided.")
@argh.arg("-d", "--logging-debug", help="If enabled, the logging level is set to `logging.DEBUG`")
def single(reference_filename: str,
           query_filename: str,
           chrlen_filename: str,
           context_filename: str = None,
           masking_filename: str = None,
           output_dir: str = f"mcdp2-results-{datetime.now().strftime('%Y%m%dT%H%M%S')}",
           ignore_bed_blocks: bool = False,
           threads: int = 4,
           statistics: str = "overlaps",
           algorithm: str = "hybrid",
           hybrid_threshold: int = 50000,
           separate_models: bool = False,
           logging_debug: bool = False):
    """Test if the reference has significantly many overlaps/shared bases with the query. Allows to use masking or context. The
    references have to be disjoint. """
    logging_level = logging.DEBUG if logging_debug else logging.INFO
    initialize_root_logger(logging_level)

    logger = logging.getLogger("mcdp2")
    logger.info(f"Starting MCDP2 for single reference test "
                f"between '{reference_filename}' and '{query_filename}...")

    create_output_directory(logger, output_dir, level=logging_level)

    logger.info(f"Loading chromosome lengths '{chrlen_filename}'...")
    with open(chrlen_filename) as f:
        chromosome_lengths = load_chromosome_lengths(f)
        logger.info(f"Loaded {len(chromosome_lengths)} chromosome lengths.")
        chromosome_names = [name for name, _ in chromosome_lengths]

    logger.info(f"Loading reference '{reference_filename}'...")
    with open(reference_filename) as f:
        reference = load_and_prune_intervals(f,
                                             chromosome_names,
                                             ignore_blocks=ignore_bed_blocks)

    logger.info(f"Loading query '{query_filename}'...")
    with open(query_filename) as f:
        query = load_and_prune_intervals(f,
                                         chromosome_names,
                                         ignore_blocks=ignore_bed_blocks)

    if context_filename is not None and masking_filename is not None:
        raise ValueError('It is not possible to specify both context and masking files! Choose '
                         'one or nothing!')
    elif context_filename is not None:
        logger.info(f"Loading context '{context_filename}'...")
        with open(context_filename) as f:
            context = load_context(f, chromosome_lengths, ignore_blocks=ignore_bed_blocks)
    elif masking_filename is not None:
        logger.info(f"Loading masking '{masking_filename}'...")
        with open(masking_filename) as f:
            masking_intervals = load_and_prune_intervals(f,
                                                         chromosome_names,
                                                         ignore_blocks=ignore_bed_blocks)
            reference = mask_intervals(intervals=reference,
                                       masking=masking_intervals)
            logger.info(f'{len(reference)} reference intervals after masking')
            query = mask_intervals(intervals=query,
                                   masking=masking_intervals)
            logger.info(f'{len(query)} query intervals after masking')
            context = create_context_from_masking(masking_intervals, chromosome_lengths, separate_models=separate_models)
    else:
        context = create_default_context(chromosome_lengths, separate_models=separate_models)

    if statistics == "overlaps":
        if algorithm == 'hybrid':
            algorithm = 'fast' if len(reference) > hybrid_threshold else 'exact'

        result = compute_pvalue_overlaps(reference=reference,
                                         query=query,
                                         chromosome_lengths=chromosome_lengths,
                                         context=context,
                                         threads=threads,
                                         algorithm=algorithm)

        with open(os.path.join(output_dir, FULL_RESULTS_OUTPUT_FILENAME), "w") as f:
            yaml.dump(result, f)

        with open(os.path.join(output_dir, SUMMARY_OUTPUT_FILENAME), "w") as f:
            write_summary_single_overlaps(f, result)
    elif statistics == "bases":
        result = compute_pvalue_bases(reference=reference,
                                         query=query,
                                         context=context,
                                         threads=threads)

        with open(os.path.join(output_dir, FULL_RESULTS_OUTPUT_FILENAME), "w") as f:
            yaml.dump(result, f)

        with open(os.path.join(output_dir, SUMMARY_OUTPUT_FILENAME), "w") as f:
            write_summary_single_bases(f, result)
    else:
        raise NotImplementedError(f"There is no statistics '{statistics}'")


def write_summary_single_overlaps(f, result):
    summary = []
    summary.append(f"Reference size: {result['total']['reference_size']} intervals")
    summary.append(f"Query size: {result['total']['query_size']} intervals")
    summary.append(f"Number of overlaps: {result['total']['overlap_count']}")
    if 'full_dp_pvalue' in result['total']:
        pvalue, zscore = result['total']['full_dp_pvalue'], result['total']['full_dp_zscore']
        summary.append(f"P-value for enrichment (exact): {pvalue}")
        summary.append(f"Z-score: {zscore}")
    else:
        pvalue, zscore = result['total']['clt_pvalue'], result['total']['clt_zscore']
        summary.append(f"P-value for enrichment (approx): {pvalue}")
        summary.append(f"Z-score: {zscore}")

    print("\n".join(summary), file=f)


def write_summary_single_bases(f, result):
    summary = []
    summary.append(f"Reference size: {result['total']['reference_size']} intervals")
    summary.append(f"Query size: {result['total']['query_size']} intervals")
    summary.append(f"Number of shared bases: {result['total']['shared_bases_count']}")
    summary.append(f"Shared coverage: {result['total']['shared_coverage']}")
    pvalue, zscore = result['total']['pvalue_sb'], result['total']['zscore_sb']
    summary.append(f"P-value for enrichment for shared bases (approx): {pvalue}")
    summary.append(f"Z-score: {zscore}")

    print("\n".join(summary), file=f)


@argh.arg("subset-filename", help="BED file with the subset reference annotation")
@argh.arg("superset-filename", help="BED file with the superset reference annotation")
@argh.arg("query-filename", help="BED file with the query annotation")
@argh.arg("chrlen-filename", help="TXT file with chromosome lengths")
@argh.arg("-o", "--output-dir", help="Output directory")
@argh.arg("--ignore-bed-blocks", help="If set, the intervals are created using chromStart and "
                                      "chromEnd")
@argh.arg("-t", "--threads", help="Number of threads to use.")
@argh.arg("-s", "--statistics", choices=["overlaps", "bases"],
          help="The test statistics to be used in the tests. 'overlaps' means the number of intervals in reference "
               "annotation hit by an interval in query. 'bases' means the number of shared bases between the "
               "reference and query annotations.")
@argh.arg("-a", "--algorithm", choices=["hybrid", "exact", "fast", 'both'],
          help="Algorithm to compute the p-value (only for '-s overlaps' option). "
               "'fast' computes a normal approximation"
               "of the total PMF and is the faster option. 'exact' computes the PMF using "
               "a quadratic DP algorithm and is slower, but it is better for smaller datasets. "
               "'hybrid' selects the algorithm based on the total size of the reference. "
               "Once the total size is greater than 'hybrid_threshold', "
               "it switches to the fast version. "
               "'both' will run both algorithms. "
               "The option is ignored if the number of shared bases is chosen as the test statistic.")
@argh.arg("--hybrid-threshold",
          help="A threshold to switch from 'exact' to 'fast' algorithm for 'hybrid' option.")
@argh.arg("--separate-models", help="If set, the individual chromosomes have separate models. "
                                    "Otherwise the underlying model is the same.")
@argh.arg("-d", "--logging-debug", help="If enabled, the logging level is set to `logging.DEBUG`")
def diff_subset(subset_filename: str,
                superset_filename: str,
                query_filename: str,
                chrlen_filename: str,
                output_dir: str = f"mcdp2-results-{datetime.now().strftime('%Y%m%dT%H%M%S')}",
                ignore_bed_blocks: bool = False,
                threads: int = 4,
                statistics: str = "overlaps",
                algorithm: str = "hybrid",
                hybrid_threshold: int = 50000,
                logging_debug: bool = False,
                separate_models: bool = False
                ):
    """Test if the subset reference has significantly more overlaps/shared bases than the superset reference."""
    logging_level = logging.DEBUG if logging_debug else logging.INFO
    initialize_root_logger(logging_level)
    logger = logging.getLogger("mcdp2")
    logger.info(f"Starting MCDP2 for subset '{subset_filename}' "
                f"and superset '{superset_filename}' reference "
                f"annotations and '{query_filename} query annotation")
    create_output_directory(logger, output_dir, level=logging_level)

    logger.info(f"Loading chromosome lengths '{chrlen_filename}'...")
    with open(chrlen_filename) as f:
        chromosome_lengths = load_chromosome_lengths(f)
        logger.info(f"Loaded {len(chromosome_lengths)} chromosome lengths.")
        chromosome_names = [name for name, _ in chromosome_lengths]

    logger.info(f"Loading subset reference '{subset_filename}'...")
    with open(subset_filename) as f:
        subset = load_and_prune_intervals(f,
                                          chromosome_names,
                                          ignore_blocks=ignore_bed_blocks)

    logger.info(f"Loading superset reference '{superset_filename}'...")
    with open(superset_filename) as f:
        superset = load_and_prune_intervals(f,
                                            chromosome_names,
                                            ignore_blocks=ignore_bed_blocks)

    logger.info(f"Loading query '{query_filename}'...")
    with open(query_filename) as f:
        query = load_and_prune_intervals(f,
                                         chromosome_names,
                                         ignore_blocks=ignore_bed_blocks)

    context = create_context_from_superset(superset, chromosome_lengths, separate_models=separate_models)

    if statistics == "overlaps":
        if algorithm == 'hybrid':
            algorithm = 'fast' if len(subset) > hybrid_threshold else 'exact'

        result = compute_pvalue_overlaps(reference=subset,
                                query=query,
                                chromosome_lengths=chromosome_lengths,
                                context=context,
                                threads=threads,
                                algorithm=algorithm)

        with open(os.path.join(output_dir, FULL_RESULTS_OUTPUT_FILENAME), "w") as f:
            yaml.dump(result, f)

        with open(os.path.join(output_dir, SUMMARY_OUTPUT_FILENAME), "w") as f:
            write_summary_subset(f, result)
    elif statistics == "bases":
        result = compute_pvalue_bases(reference=subset,
                                         query=query,
                                         context=context,
                                         threads=threads)

        with open(os.path.join(output_dir, FULL_RESULTS_OUTPUT_FILENAME), "w") as f:
            yaml.dump(result, f)

        with open(os.path.join(output_dir, SUMMARY_OUTPUT_FILENAME), "w") as f:
            write_summary_subset_bases(f, result)
    else:
        raise NotImplementedError(f"There is no statistics '{statistics}'")


def write_summary_subset(f, result):
    summary = []
    summary.append(f"Subset reference size: {result['total']['reference_size']} intervals")
    summary.append(f"Query size: {result['total']['query_size']} intervals")
    summary.append(f"Number of overlaps: {result['total']['overlap_count']}")
    if 'full_dp_pvalue' in result['total']:
        pvalue, zscore = result['total']['full_dp_pvalue'], result['total']['full_dp_zscore']
        summary.append(f"P-value for enrichment (exact): {pvalue}")
        summary.append(f"Z-score: {zscore}")
    else:
        pvalue, zscore = result['total']['clt_pvalue'], result['total']['clt_zscore']
        summary.append(f"P-value for enrichment (approx): {pvalue}")
        summary.append(f"Z-score: {zscore}")

    print("\n".join(summary), file=f)


def write_summary_subset_bases(f, result):
    summary = []
    summary.append(f"Subset reference size: {result['total']['reference_size']} intervals")
    summary.append(f"Query size: {result['total']['query_size']} intervals")
    summary.append(f"Number of shared bases: {result['total']['shared_bases_count']}")
    summary.append(f"Shared coverage: {result['total']['shared_coverage']}")
    pvalue, zscore = result['total']['pvalue_sb'], result['total']['zscore_sb']
    summary.append(f"P-value for enrichment for shared bases (approx): {pvalue}")
    summary.append(f"Z-score: {zscore}")

    print("\n".join(summary), file=f)


@argh.arg("ref1-filename", help="BED file with the first reference annotation")
@argh.arg("ref2-filename", help="BED file with the second reference annotation")
@argh.arg("query-filename", help="BED file with the query annotation")
@argh.arg("chrlen-filename", help="TXT file with chromosome lengths")
@argh.arg("-o", "--output-dir", help="Output directory")
@argh.arg("--ignore-bed-blocks", help="If set, the intervals are created using chromStart and "
                                      "chromEnd")
@argh.arg("-t", "--threads", help="Number of threads to use.")
@argh.arg("-s", "--statistics", choices=["overlaps", "bases"],
          help="The test statistics to be used in the tests. 'overlaps' means the number of intervals in reference "
               "annotation hit by an interval in query. 'bases' means the number of shared bases between the "
               "reference and query annotations.")
@argh.arg("-a", "--algorithm", choices=["hybrid", "exact", "fast", 'both'],
          help="Algorithm to compute the p-value (only for '-s overlaps' option). "
               "'fast' computes a normal approximation"
               "of the total PMF and is the faster option. 'exact' computes the PMF using "
               "a quadratic DP algorithm and is slower, but it is better for smaller datasets. "
               "'hybrid' selects the algorithm based on the total size of the reference. "
               "Once the total size is greater than 'hybrid_threshold', "
               "it switches to the fast version. "
               "'both' will run both algorithms. "
               "The option is ignored if the number of shared bases is chosen as the test statistic.")
@argh.arg("--separate-models", help="If set, the individual chromosomes have separate models. "
                                    "Otherwise the underlying model is the same.")
@argh.arg("--hybrid-threshold",
          help="A threshold to switch from 'exact' to 'fast' algorithm for 'hybrid' option.")
@argh.arg("-d", "--logging-debug", help="If enabled, the logging level is set to `logging.DEBUG`")
def diff_disjoint(ref1_filename: str,
                  ref2_filename: str,
                  query_filename: str,
                  chrlen_filename: str,
                  output_dir: str = f"mcdp2-results-{datetime.now().strftime('%Y%m%dT%H%M%S')}",
                  ignore_bed_blocks: bool = False,
                  threads: int = 4,
                  algorithm: str = "hybrid",
                  statistics: str = "overlaps",
                  hybrid_threshold: int = 50000,
                  logging_debug: bool = False,
                  separate_models: bool = False
                  ):
    """Test if the first reference has significantly more overlaps/shared bases than the second one."""
    logging_level = logging.DEBUG if logging_debug else logging.INFO
    initialize_root_logger(logging_level)

    logger = logging.getLogger("mcdp2")
    logger.info(f"Starting MCDP2 for '{ref1_filename}' "
                f"and '{ref2_filename}' reference "
                f"annotations and '{query_filename} query annotation")
    create_output_directory(logger, output_dir, level=logging_level)

    logger.info(f"Loading chromosome lengths '{chrlen_filename}'...")
    with open(chrlen_filename) as f:
        chromosome_lengths = load_chromosome_lengths(f)
        logger.info(f"Loaded {len(chromosome_lengths)} chromosome lengths.")
        chromosome_names = [name for name, _ in chromosome_lengths]

    logger.info(f"Loading first reference '{ref1_filename}'...")
    with open(ref1_filename) as f:
        ref1 = load_and_prune_intervals(f,
                                        chromosome_names,
                                        ignore_blocks=ignore_bed_blocks)

    logger.info(f"Loading second reference '{ref2_filename}'...")
    with open(ref2_filename) as f:
        ref2 = load_and_prune_intervals(f,
                                        chromosome_names,
                                        ignore_blocks=ignore_bed_blocks)

    logger.info(f"Loading query '{query_filename}'...")
    with open(query_filename) as f:
        query = load_and_prune_intervals(f,
                                         chromosome_names,
                                         ignore_blocks=ignore_bed_blocks)

    total_ref = merge_nondisjoint_intervals(ref1 + ref2)

    context = create_context_from_superset(total_ref, chromosome_lengths, separate_models=separate_models)

    if statistics == "overlaps":
        if algorithm == 'hybrid':
            algorithm = 'fast' if len(ref1) > hybrid_threshold else 'exact'

        result = compute_pvalue_overlaps(reference=ref1,
                                query=query,
                                chromosome_lengths=chromosome_lengths,
                                context=context,
                                threads=threads,
                                algorithm=algorithm)

        with open(os.path.join(output_dir, FULL_RESULTS_OUTPUT_FILENAME), "w") as f:
            yaml.dump(result, f)

        with open(os.path.join(output_dir, SUMMARY_OUTPUT_FILENAME), "w") as f:
            write_summary_disjoint(f, result)

    elif statistics == "bases":
        result = compute_pvalue_bases(reference=ref1,
                                         query=query,
                                         context=context,
                                         threads=threads)

        with open(os.path.join(output_dir, FULL_RESULTS_OUTPUT_FILENAME), "w") as f:
            yaml.dump(result, f)

        with open(os.path.join(output_dir, SUMMARY_OUTPUT_FILENAME), "w") as f:
            write_summary_disjoint_bases(f, result)
    else:
        raise NotImplementedError(f"There is no statistics '{statistics}'")


def write_summary_disjoint(f, result):
    summary = []
    summary.append(f"First reference size: {result['total']['reference_size']} intervals")
    summary.append(f"Query size: {result['total']['query_size']} intervals")
    summary.append(f"Number of overlaps: {result['total']['overlap_count']}")
    if 'full_dp_pvalue' in result['total']:
        pvalue, zscore = result['total']['full_dp_pvalue'], result['total']['full_dp_zscore']
        summary.append(f"P-value for enrichment (exact): {pvalue}")
        summary.append(f"Z-score: {zscore}")
    else:
        pvalue, zscore = result['total']['clt_pvalue'], result['total']['clt_zscore']
        summary.append(f"P-value for enrichment (approx): {pvalue}")
        summary.append(f"Z-score: {zscore}")

    print("\n".join(summary), file=f)


def write_summary_disjoint_bases(f, result):
    summary = []
    summary.append(f"First reference size: {result['total']['reference_size']} intervals")
    summary.append(f"Query size: {result['total']['query_size']} intervals")
    summary.append(f"Number of shared bases: {result['total']['shared_bases_count']}")
    summary.append(f"Shared coverage: {result['total']['shared_coverage']}")
    pvalue, zscore = result['total']['pvalue_sb'], result['total']['zscore_sb']
    summary.append(f"P-value for enrichment for shared bases (approx): {pvalue}")
    summary.append(f"Z-score: {zscore}")

    print("\n".join(summary), file=f)


def initialize_root_logger(level=logging.DEBUG):
    logger = logging.getLogger("mcdp2")
    logger.setLevel(level=level)
    logger.handlers = []
    add_console_handler(logger, level=level)


def create_output_directory(logger, output_dir, level=logging.DEBUG):
    if os.path.exists(output_dir) and os.path.isfile(output_dir):
        raise IOError(f"The output directory '{output_dir}' already exists "
                      f"and it is actually a file! Change the name or "
                      f"delete the existing file!")
    elif not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log_filename = os.path.join(output_dir, LOG_FILENAME)
    add_file_handler(logger, log_filename, level=level)


def load_and_prune_intervals(f, chromosome_names, ignore_blocks):
    logger = logging.getLogger("mcdp2")
    raw_intervals = load_intervals_from_bed(f, ignore_blocks=ignore_blocks)
    raw_count = len(raw_intervals)
    logger.info(f"Loaded {raw_count} raw intervals.")

    intervals_on_genome = filter_intervals_by_chrnames(raw_intervals, chromosome_names)
    on_genome_count = len(intervals_on_genome)
    logger.info(f"{on_genome_count} of loaded intervals are on the genome.")

    disjoint_intervals = merge_nondisjoint_intervals(intervals_on_genome)
    disjoint_count = len(disjoint_intervals)
    logger.info(f"{disjoint_count} disjoint intervals after merging.")
    return disjoint_intervals


def add_console_handler(logger, level=logging.DEBUG):
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def add_file_handler(logger, log_filename, level):
    fileHandler = logging.FileHandler(log_filename)
    fileHandler.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)


def main():
    argh.dispatch_commands([single, diff_subset, diff_disjoint])


if __name__ == "__main__":
    main()
