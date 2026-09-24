"""Export separate manuscript panels, text-free versions and legends."""
import argparse
from pathlib import Path

from figures import manuscript as plots
from figures.common import HIGH, OLD, read_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('figure', choices=['2', '3', '4', '5A', '5BCD', 'S1', 'S2', 'S3', 'S4',
                                          'S5', 'S6', 'S7', 'S8', 'S9', 'S10'])
    parser.add_argument('--input', required=True, type=Path,
                        help='Input workbook, summary table or simulation output directory.')
    parser.add_argument('--other', type=Path, help='All-or-nothing hybrid directory for S4.')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--formats', nargs='+', choices=['png', 'pdf', 'svg'], default=['png', 'pdf'])
    args = parser.parse_args()
    if not args.input.exists():
        parser.error(f'Input does not exist: {args.input}')
    if args.figure == 'S4' and (args.other is None or not args.other.is_dir()):
        parser.error('S4 requires --other pointing to the all-or-nothing hybrid results.')
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Choose an empty or new figure output directory.')
    output.mkdir(parents=True, exist_ok=True)
    number = args.figure
    if number == '2':
        plots.network_panels(args.input, output, args.formats)
    elif number in ['3', '4', 'S1', 'S2']:
        plots.heatmaps(read_summary(args.input), HIGH if number in ['3', 'S1'] else OLD,
                       'Fig'+number, output, args.formats)
    elif number in ['5A', 'S3']:
        plots.endpoint_map(read_summary(args.input), output, args.formats, number == 'S3')
    elif number == '5BCD':
        plots.hybrid_panels(args.input, output, args.formats)
    elif number == 'S4':
        plots.comparison(args.input, args.other, output, args.formats)
    elif number == 'S5':
        plots.coverage_panels(read_summary(args.input), output, args.formats)
    elif number in ['S6', 'S7']:
        plots.disease_panels(args.input, output, args.formats, HIGH if number == 'S6' else OLD)
    elif number == 'S8':
        plots.network_sensitivity(args.input, output, args.formats)
    elif number == 'S9':
        plots.uncertainty_panels(read_summary(args.input), output, args.formats)
    else:
        plots.representative_intervals(read_summary(args.input), output, args.formats)
    print(f'Exported Figure {number} panels to {output}')


if __name__ == '__main__':
    main()
