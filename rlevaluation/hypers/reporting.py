import numpy as np
from rlevaluation.hypers.api import HyperSelectionResult
from rlevaluation.config import DataDefinition, maybe_global


def pretty_print(result: HyperSelectionResult, d: DataDefinition | None = None):
    if isinstance(result, HyperSelectionResult):
        return pretty_print_hyper_selection_result(result, d)

    raise Exception(f'Cannot pretty print <{type(result)}>')


def pretty_print_hyper_selection_result(result: HyperSelectionResult, d: DataDefinition | None = None):
    d = maybe_global(d)

    cols = result.config_params
    col_len = max(map(len, cols))

    out = ''

    # best hypers
    out += 'Best configuration setting:\n'
    out += '---------------------------\n'
    for hyper, value in zip(cols, result.best_configuration, strict=True):
        if isinstance(value, float) and np.isnan(value): continue
        ws = 4 + col_len - len(hyper)
        out += f'{hyper}:' + ' ' * ws
        out += f'{value}\n'

    out += '\n'
    out += f'Score for best setting: {result.best_score:.3f}\n'
    out += f'Confidence interval: ({result.ci[0]:.3f}, {result.ci[1]:.3f})\n'
    out += '---------------------------\n\n'

    if len(result.uncertainty_set_probs) > 1:
        out += 'Possible best configurations:\n'
        out += '-----------------------------\n'
        for i, hyper in enumerate(cols):
            hyper_val = result.uncertainty_set_configurations[0][i]
            if isinstance(hyper_val, float) and np.isnan(hyper_val): continue
            ws = 4 + col_len - len(hyper)
            out += f'{hyper}:' + ' ' * ws

            for config in result.uncertainty_set_configurations:
                out += f'{config[i]}  '

            out += '\n'

        out += '\n'
        out += f'With probabilities: {result.uncertainty_set_probs}\n'
        out += f'Score over best configurations: {result.sample_stat:.3f}\n'

    print(out)
