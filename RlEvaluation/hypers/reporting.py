import numpy as np
from RlEvaluation.hypers.api import HyperSelectionResult
from RlEvaluation.config import DataDefinition, maybe_global


def pretty_print(result: HyperSelectionResult, d: DataDefinition | None = None):
    if isinstance(result, HyperSelectionResult):
        return pretty_print_hyper_selection_result(result, d)

    raise Exception(f'Cannot pretty print <{type(result)}>')


def pretty_print_hyper_selection_result(result: HyperSelectionResult, d: DataDefinition | None = None):
    d = maybe_global(d)

    col_len = max(map(len, d.hyper_cols))

    out = ''

    # best hypers
    out += 'Best configuration setting:\n'
    out += '---------------------------\n'
    for hyper, value in zip(d.hyper_cols, result.best_configuration):
        if np.isnan(value): continue
        ws = 4 + col_len - len(hyper)
        out += f'{hyper}:' + ' ' * ws
        out += f'{value}\n'

    out += '\n'
    out += f'Score for best setting: {result.best_score:.3f}\n'
    out += '---------------------------\n\n'

    if len(result.uncertainty_set_probs) > 1:
        out += 'Possible best configurations:\n'
        out += '-----------------------------\n'
        for i, hyper in enumerate(d.hyper_cols):
            if np.isnan(result.uncertainty_set_configurations[0][i]): continue
            ws = 4 + col_len - len(hyper)
            out += f'{hyper}:' + ' ' * ws

            for config in result.uncertainty_set_configurations:
                out += f'{config[i]}  '

            out += '\n'

        out += '\n'
        out += f'With probabilities: {result.uncertainty_set_probs}\n'
        out += f'Score over best configurations: {result.sample_stat:.3f}\n'
        out += f'Confidence interval: ({result.ci[0]:.3f}, {result.ci[1]:.3f})'

    print(out)
