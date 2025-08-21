## 1.1.2 (2025-08-21)

### Fix

- array empty check for numpy >= 2.2

## 1.1.1 (2025-02-21)

### Fix

- return hyper->value dict from hyperparam search

## 1.1.0 (2025-02-18)

### Feat

- add tolerance_interval to the public api
- add public method to get tolerance interval for set of samples

### Fix

- remove old bootstrap ci methods

## 1.0.0 (2025-02-17)

### BREAKING CHANGE

- the external API previously relied on pandas for data
format. Now the expected data format is a polars dataframe.
- with this set of changes, we will fundamentally
rework the public API for this library. This new API will be more stable
going forward, so we should be ready to bump this to v1.

### Feat

- allow columns to be a subset
- add tolerance intervals
- grab learning curves for multiple hypers
- add ability to compute step-weighted return
- add ability to compute step_return
- add simulation-based hyper selection
- make configuration a global dataclass

### Fix

- typo building logger class
- use modern numpy rng
- allow env and alg strings to be null
- ensure consistent types
- don't require alg and env columns
- be defensive about numpy types
- harden interpolation assumptions (such as monotonicity of time)
- check for na in first column
- use episode length for weighting
- ignore nans in a metric
- add type-check before isnan
- only use observed max_steps for reweighting

### Refactor

- complete transition to polars over pandas
- migrate extract_learning_curves to polars
- move hyper evaluation to polars
- add score data format utilities
- rename base directory as rlevaluation
- change import path to rlevaluation
- move numba functions into backend
- remove now outdated interfaces

## 0.1.1 (2023-05-15)

### Fix

- allow access to underlying dataframe

## 0.1.0 (2023-05-15)

### Feat

- rework external api
- hyper selection should return idx

### Fix

- use modern versions of numpy as numba
- handle ragged results lists

## 0.0.3 (2023-03-03)

### Fix

- make modules visible
- mark projected as typed

## 0.0.2 (2023-03-03)

### Fix

- fix picky type error

## 0.0.1 (2023-03-02)

### Fix

- firm up public facing types
