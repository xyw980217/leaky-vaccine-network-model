"""Check source portability and, optionally, network and allocation invariants."""
import argparse
import ast
import importlib
import os
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path)
    args = parser.parse_args()
    files = list(ROOT.rglob('*.py'))
    for path in files:
        source = path.read_bytes().decode('ascii')
        ast.parse(source, filename=str(path))
        assert ':\\Users\\' not in source and ':\\Anaconda' not in source, path
    print(f'PASS: {len(files)} Python files parse and contain ASCII source only.')
    if args.data is None:
        return
    data = args.data.resolve()
    if not data.is_file():
        parser.error(f'Workbook not found: {data}')
    sys.path[:0] = [str(ROOT), str(ROOT/'simulations')]
    with tempfile.TemporaryDirectory(prefix='leaky_checks_') as temporary:
        os.environ.update(LEAKY_DATA=str(data), LEAKY_OUTPUT=temporary, LEAKY_RUNS='2',
                          LEAKY_DAYS='20', LEAKY_WORKERS='1', LEAKY_R0='1',
                          LEAKY_VE='0.8', LEAKY_ALPHA='0,0.5,1')
        import numpy as np
        import networkx as nx
        from figures._network_preparation import rebuild_network
        graph, nodes, _, _ = rebuild_network(data)
        from parameter_grid import recipient_mask, STRATEGIES
        primary = importlib.import_module('_primary')
        np.testing.assert_array_equal(primary.total_node_strength, nodes['Total_Weighted_Exposure_Strength'])
        assert primary.G.number_of_edges() == graph.number_of_edges()
        oldest = recipient_mask(primary, STRATEGIES[2], .5)
        high = recipient_mask(primary, STRATEGIES[1], .5)
        shared = oldest & high
        for name in ['hybrid_leaky', 'hybrid_all_or_nothing']:
            model = importlib.import_module(name)
            np.testing.assert_array_equal(model.get_hybrid_vaccinated_mask(0, .5), oldest)
            np.testing.assert_array_equal(model.get_hybrid_vaccinated_mask(1, .5), high)
            for alpha in np.linspace(0, 1, 51):
                mask, _, info = model.get_quota_mixture_allocation(alpha, .5)
                assert mask.sum() == int(.5*model.N) and mask[shared].all()
                slots = int(high.sum()-shared.sum())
                assert info['High Exposure Exclusive Quota'] == int(np.floor(alpha*slots+.5))
            mask = model.get_hybrid_vaccinated_mask(.5, .5)
            beta = model.calibrate_scenario(1)[0]
            np.random.seed(42)
            inputs = [beta, mask, .8]
            if name == 'hybrid_all_or_nothing':
                inputs.append(model.get_aon_protected_mask(mask, .8, 1., .5, 1)[1])
            result = model.run_strategy_simulation(*inputs)
            assert result[8].sum() == result[0]
            assert result[9].sum() == result[1]
            assert 0 <= result[1] <= result[0] <= model.N
            print(f'PASS: {name}, 51 allocations and endpoint group totals.')
        print(f'Network: {len(graph)} nodes, {graph.number_of_edges()} edges, '
              f'{len(max(nx.connected_components(graph), key=len))} nodes in the largest component.')
        print(f'Recipients: {high.sum()} per strategy, {shared.sum()} shared.')


if __name__ == '__main__':
    main()
