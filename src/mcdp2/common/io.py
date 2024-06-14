import re
import sys
from collections import defaultdict
from enum import IntEnum


def load_context(fd, chromosome_lengths, ignore_blocks):
    _, raw_records = _parse_bed_intervals(fd)
    raw_context_intervals = _make_context_intervals_from_records(raw_records, ignore_blocks)
    context = _fill_background_context(raw_context_intervals, chromosome_lengths)
    return context


def _make_context_intervals_from_records(records, ignore_blocks):
    result = defaultdict(list)
    for record in records:
        chrom = record[0]
        chromStart = record[1]
        chromEnd = record[2]
        context_name = record[3]
        if len(record) >= 12 and (not ignore_blocks):
            block_count = record[9]
            block_sizes = record[10]
            block_starts = record[11]
            for i in range(block_count):
                result[chrom].append((context_name,
                                      chromStart + block_starts[i],
                                      chromStart + block_starts[i] + block_sizes[i]))
        else:
            result[chrom].append((context_name, chromStart, chromEnd))
    return dict(result)


def _fill_background_context(raw_intervals, chromosome_lengths):
    BACKGROUND_SYMBOL = "__B"
    result = {name: [] for name, length in chromosome_lengths}
    for name, length in chromosome_lengths:
        genome_intervals = sorted(raw_intervals.get(name, []), key=lambda x: (x[1], x[2]))
        for ctx, b, e in genome_intervals:
            if len(result[name]) == 0:
                if b == 0:
                    result[name].append((ctx, b, e))
                else:
                    result[name].append((BACKGROUND_SYMBOL, 0, b))
                    result[name].append((ctx, b, e))
            else:
                prev_ctx, prev_b, prev_e = result[name][-1]
                if prev_e < b:
                    result[name].append((BACKGROUND_SYMBOL, prev_e, b))
                    result[name].append((ctx, b, e))
                elif prev_e == b and prev_ctx != ctx:
                    result[name].append((ctx, b, e))
                elif prev_e >= b and prev_ctx == ctx:
                    result[name][-1] = (ctx, prev_b, max(prev_e, e))
                else:
                    # overlap of intervals with different contexts
                    raise ValueError("Context intervals with different names cannot overlap!"
                                     f" (chromosome {name}, interval [{b}, {e}); {prev_ctx=}, {prev_b=}, {prev_e=})")
        if len(result[name]) == 0:
            result[name].append((BACKGROUND_SYMBOL, 0, length))
        elif (last := result[name][-1][2]) < length:
            result[name].append((BACKGROUND_SYMBOL, last, length))
        elif result[name][-1][2] == length:
            pass  # nothing to do here
        else:
            # overflow of the genome length
            raise ValueError(f"Context intervals should not overflow "
                             f"the original genome length! (chromosome {name})")
    return result


def create_default_context(chromosome_lengths, separate_models=False):
    result = {}
    for name, length in chromosome_lengths:
        if separate_models:
            result[name] = [(name, 0, length)]
        else:
            result[name] = [("g", 0, length)]
    return result


def load_chromosome_lengths(fd):
    """Load chromosome lengths from a .genome file."""
    results = []
    for linenum, line in enumerate(fd):
        line = line.strip()
        if len(line) == 0:
            continue
        elements = line.split("\t")
        if len(elements) != 2:
            raise ValueError(f'Line {linenum+1}: a line should contain two tab-separated columns!')

        name = elements[0]
        try:
            length = int(elements[1])
        except ValueError:
            raise ValueError(f'Line {linenum+1}: a genome length should be a positive integer!')
        if length <= 0:
            raise ValueError(f'Line {linenum+1}: a genome length should be a positive integer!')

        results.append((name, length))

    # check for duplicate names
    if len(set(n for n, l in results)) < len(results):
        raise ValueError(f'Duplicate chromosome names are detected! Please remove them!')
    return results


def load_intervals_from_bed(fd, ignore_blocks=False):
    """Loads intervals from a BED file. If `ignore_blocks` is `True`, then
    returns the intervals (chromStarts, chromEnds).
    if `ignore_blocks` is `False`, then splits the records into individual blocks."""
    _, records = _parse_bed_intervals(fd)
    if ignore_blocks:
        results = [(r[0], r[1], r[2]) for r in records]
    else:
        results = []
        for r in records:
            track = r[0]
            chrom_start = r[1]
            if len(r) < 10:
                results.append((track, chrom_start, r[2]))
                continue
            block_count = r[9]
            sizes = r[10]
            starts = r[11]
            for i in range(block_count):
                results.append((track, chrom_start + starts[i], chrom_start + starts[i] + sizes[i]))
    return results


BED_HEADER_PREFIXES = ["browser", "track", "#"]
BED_HEADER_PATTERN = re.compile("^(" + "|".join(BED_HEADER_PREFIXES) + ")")
BED_SEPARATOR = "\t"


def _is_bed_header_line(line: str) -> bool:
    return BED_HEADER_PATTERN.match(line) is not None


def _parse_bed_intervals(fd):
    """Parse a BED file.
    Format is described at https://en.wikipedia.org/wiki/BED_(file_format).

    Returns two lists: list of header lines and a list of records
    """
    header_lines = []
    records = []
    for line_num, line in enumerate(fd):
        line = line.strip()
        # skip empty lines
        if len(line) == 0:
            continue

        # check if the line is a header line
        if _is_bed_header_line(line):
            header_lines.append(line)
            continue

        elements = line.split(BED_SEPARATOR)
        if not 3 <= len(elements) <= 12:
            raise ValueError(f'Line {line_num+1}: A record in BED file should have between '
                             f'3 and 12 columns, found {len(elements)} instead!')
        record = []
        # 1	chrom	Chromosome (e.g. chr3, chrY, chr2_random) or scaffold
        # (e.g. scaffold10671) name
        record.append(elements[0])

        # 2	chromStart	Start coordinate on the chromosome or scaffold for the sequence considered
        # (the first base on the chromosome is numbered 0)
        try:
            value = int(elements[1])
        except ValueError:
            raise ValueError(f'Line {line_num+1}: Second column "chromStart" must contain an non-negative integer!')
        if value < 0:
            raise ValueError(f'Line {line_num+1}: Second column "chromStart" must contain a non-negative integer!')
        record.append(value)

        # 3	chromEnd	End coordinate on the chromosome or scaffold for the sequence considered. This position is non-inclusive, unlike chromStart.
        try:
            value = int(elements[2])
        except ValueError:
            raise ValueError(f'Line {line_num+1}: Third column "chromEnd" must contain a non-negative integer!')
        if value < 0:
            raise ValueError(f'Line {line_num+1}: Third column "chromEnd" must contain a non-negative integer!')
        record.append(value)

        # 4	name	Name of the line in the BED file
        if len(elements) >= 4:
            record.append(elements[3])

        # 5	score	Score between 0 and 1000
        if len(elements) >= 5:
            try:
                value = int(elements[4])
            except ValueError:
                raise ValueError(f'Line {line_num+1}: Fifth column "score" must contain an integer between 0 and 1000!')
            if value not in range(1001):
                raise ValueError(f'Line {line_num+1}: Fifth column "score" must contain an integer between 0 and 1000!')
            record.append(value)

        # 6	strand	DNA strand orientation (positive ["+"] or negative ["-"] or "." if no strand)
        if len(elements) >= 6:
            value = elements[5]
            if value not in ("+", "-", "."):
                raise ValueError(f'Line {line_num + 1}: Sixth column "strand" must contain either "+", "-" or "."!')
            record.append(value)

        # 7	thickStart	Starting coordinate from which the annotation is displayed in a thicker way on a graphical representation (e.g.: the start codon of a gene)
        if len(elements) >= 7:
            if len(elements) < 8:
                raise ValueError(f'Line {line_num+1}: if the record has at least 7 columns, it should contain at least 8 columns!')
            try:
                value = int(elements[6])
            except ValueError:
                raise ValueError(
                    f'Line {line_num + 1}: Seventh column "thickStart" must contain a non-negative integer!')
            if value < 0:
                raise ValueError(
                    f'Line {line_num + 1}: Seventh column "thickStart" must contain a non-negative integer!')
            record.append(value)

            try:
                value = int(elements[7])
            except ValueError:
                raise ValueError(
                    f'Line {line_num + 1}: Eighth column "thickEnd" must contain a non-negative integer!')
            if value < 0:
                raise ValueError(
                    f'Line {line_num + 1}: Eighth column "thickEnd" must contain a non-negative integer!')
            record.append(value)

        # 9	itemRgb	RGB value in the form R,G,B (e.g. 255,0,0) determining the display color of the annotation contained in the BED file
        if len(elements) >= 9:
            value = elements[8]
            components = value.split(",")
            if len(components) != 3:
                raise ValueError(f'Line {line_num+1}: Ninth column "itemRgb" should contain three '
                                 f'comma-separated integers from range 0 to 255!')
            try:
                r, g, b = (int(x) for x in components)
            except ValueError:
                raise ValueError(f'Line {line_num+1}: Ninth column "itemRgb" should contain three '
                                 f'comma-separated integers from range 0 to 255!')

            if r not in range(256) or g not in range(256) or b not in range(256):
                raise ValueError(f'Line {line_num+1}: Ninth column "itemRgb" should contain three '
                                 f'comma-separated integers from range 0 to 255!')
            record.append([r, g, b])

        # 10	blockCount	Number of blocks (e.g. exons) on the line of the BED file
        # 11	blockSizes	List of values separated by commas corresponding to the size of the blocks (the number of values must correspond to that of the "blockCount")
        # 12	blockStarts	List of values separated by commas corresponding to the starting coordinates of the blocks, coordinates calculated relative to those present in the chromStart column (the number of values must correspond to that of the "blockCount")
        if len(elements) >= 10:
            if len(elements) < 12:
                raise ValueError(f'Line {line_num+1}: If record contains 10 columns, it should also contain 11th and 12th columns with block sizes!')
            try:
                value = int(elements[9])
            except ValueError:
                raise ValueError(f'Line {line_num+1}: Tenth column "blockCount" should contain a positive integer!')
            if value <= 0:
                raise ValueError(
                    f'Line {line_num + 1}: Tenth column "blockCount" should contain a positive integer!')
            record.append(value)

            sizes_raw = elements[10].split(",")
            if sizes_raw[-1] == "":
                sizes_raw = sizes_raw[:-1]
            if len(sizes_raw) != record[9]:
                raise ValueError(f'Line {line_num+1}: Eleventh column "blockSizes" should contain '
                                 f'as many elements as is written in the tenth column "blockCount"!')
            try:
                sizes = [int(x) for x in sizes_raw]
            except ValueError:
                raise ValueError(f'Line {line_num+1}: Eleventh column "blockSizes" should contain comma-separated non-negatiive integers!')
            if any(x < 0 for x in sizes):
                raise ValueError(f'Line {line_num+1}: Eleventh column "blockSizes" should contain comma-separated non-negatiive integers!')
            if sum(sizes) > record[2] - record[1]:
                raise ValueError(f'Line {line_num+1}: Block sizes are bigger than the record itself!')
            record.append(sizes)

            starts_raw = elements[11].split(",")
            if starts_raw[-1] == "":
                starts_raw = sizes_raw[:-1]
            if len(starts_raw) != record[9]:
                raise ValueError(
                    f'Line {line_num + 1}: Twelfth column "blockStarts" should contain '
                    f'as many elements as is written in the tenth column "blockCount"!')
            try:
                starts = [int(x) for x in starts_raw]
            except ValueError:
                raise ValueError(
                    f'Line {line_num + 1}: Twelfth column "blockStarts" should contain comma-separated non-negatiive integers!')
            if any(x < 0 for x in starts):
                raise ValueError(
                    f'Line {line_num + 1}: Twelfth column "blockStarts" should contain comma-separated non-negatiive integers!')
            if any(i > j for i, j in zip(starts[:-1], starts[1:])):
                raise ValueError(f'Line {line_num+1}: Block starts should be non-decreasing!')
            if any(x >= record[2] - record[1] for x in starts):
                raise ValueError(f'Line {line_num+1}: Block starts cannot exceed the record total length!')
            for i in range(record[9]-1):
                size = sizes[i]
                start = starts[i]
                end = start + size
                if end > starts[i+1]:
                    raise ValueError(f'Line {line_num+1}: Blocks should not overlap!')
            record.append(starts)
        records.append(tuple(record))

    return header_lines, records


def create_context_from_superset(superset, chromosome_lengths, separate_models=False):
    raw_context_intervals = {chrname: [] for chrname, _ in chromosome_lengths}
    for c, b, e in superset:
        if separate_models:
            raw_context_intervals[c].append((f'{c}_superset', b, e))
        else:
            raw_context_intervals[c].append((f'superset', b, e))

    context = _fill_background_context(raw_context_intervals, chromosome_lengths)
    return context


def create_context_from_masking(masking_intervals, chromosome_lengths, separate_models=False):
    class Event(IntEnum):
        END = 0
        BEGIN = 1
        CBEGIN = -2
        CEND = -1
    events = []
    for c, b, e in masking_intervals:
        events.append((c, b, Event.BEGIN))
        events.append((c, e, Event.END))
    for c, l in chromosome_lengths.items():
        events.append((c, 0, Event.CBEGIN))
        events.append((c, l, Event.CEND))

    result = {c: [] for c in chromosome_lengths}
    prev_position = None

    print(events, file=sys.stderr)

    for c, pos, e in sorted(events):
        match e:
            case Event.END:
                if separate_models:
                    result[c].append((f"{c}_masked", prev_position, pos))
                else:
                    result[c].append((f"masked", prev_position, pos))
            case Event.BEGIN:
                if separate_models:
                    result[c].append((f"{c}_unmasked", prev_position, pos))
                else:
                    result[c].append((f"unmasked", prev_position, pos))
            case Event.CEND:
                if separate_models:
                    result[c].append((f"{c}_unmasked", prev_position, pos))
                else:
                    result[c].append((f"unmasked", prev_position, pos))
        prev_position = pos
    return result
