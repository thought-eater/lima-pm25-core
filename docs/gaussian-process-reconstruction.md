# Gaussian Process Reconstruction

## Scientific Role

`lima_pm25.estimation` reconstructs the current PM2.5 field from sparse public
sensor measurements. The posterior mean is the R2.1 concentration estimate and
the posterior standard deviation is the associated R2.2 model uncertainty. Only
sensor locations contain measurements; values elsewhere are estimates and must
not be represented as observed ground truth.

The estimator consumes `MeasurementRecord` values, which expose observed PM2.5,
event-time position, UTC timestamp, declared random-noise standard deviation,
quality, and provenance. Hidden simulator truth and pre-error values are not
part of the reconstruction request.

## Model

The covariance is:

```text
ConstantKernel(prior_variance)
    * Matern(length_scale=[x_m, y_m, time_s], nu=1.5)
```

Coordinates are centered on the grid's southwest bounds but remain in projected
meters. Time is represented in seconds relative to the requested reconstruction
instant; prediction cells use time zero and past observations use negative
relative times. Concentration residuals are fitted relative to the configured
constant prior mean, so a prediction far from observations approaches the prior
mean and prior standard deviation.

All length scales are explicit and remain fixed during a reconstruction. Their
selection and limitations are documented in
[historical calibration](historical-calibration.md). The operational profile is
selected using calibration data before retained synthetic evaluation.

## Temporal Aging

An observation remains an exact record of its original time. Its ability to
describe a later field decreases as the difference between its timestamp and
the reconstruction timestamp grows relative to `temporal_length_scale_s`.
Consequently, uncertainty is practically zero at the exact space-time support
of a zero-noise observation and grows toward prior uncertainty at later times.

Every accepted observation supplied in a `ReconstructionRequest` and available
at or before reconstruction time is used. The caller may provide a deterministic
subset appropriate to the scenario; the reconstructor does not impose another
implicit trailing window, maximum sample count, or age-based deletion. Temporal
relevance within the supplied set is represented by the kernel.

## Observation Noise

The baseline simulated sensors declare zero bias and zero random noise. For each
accepted observation, the regressor receives:

```text
observation_variance = noise_stddev_ug_m3 ** 2
numerical_jitter = 1e-10 * prior_stddev_ug_m3 ** 2
alpha = observation_variance + numerical_jitter
```

The scale-relative jitter only stabilizes the covariance matrix and is recorded
separately from physical sensor noise. It is an internal numerical safeguard,
not a scenario parameter. There is no `WhiteKernel`, because a global residual-
noise floor would prevent posterior uncertainty from approaching zero at an
exact baseline observation. Future explicitly configured random sensor noise is
represented by the per-observation `alpha`; systematic bias is not reinterpreted
as variance.
The relative factor `1e-10` follows the scale of the kernel diagonal and remains
small enough not to materially replace a declared zero-noise observation with a
noisy one.

Numeric records marked `CLIPPED` are excluded by default because a reported
bound is not necessarily an exact concentration. Set
`include_clipped_observations=True` only to treat those bounds as ordinary
numeric approximations; this does not implement a censored-observation
likelihood. Observation metadata counts clipped records excluded by this policy
separately from dropped attempts.

## Usage

```python
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionConfig,
    ReconstructionRequest,
)

config = ReconstructionConfig(
    prior_mean_ug_m3=20.0,
    prior_stddev_ug_m3=15.0,
    x_length_scale_m=1_000.0,
    y_length_scale_m=1_000.0,
    temporal_length_scale_s=3_600.0,
    include_clipped_observations=False,
)
reconstructor = GaussianProcessFieldReconstructor(
    config,
    prediction_chunk_size=2_048,
)
posterior = reconstructor.reconstruct(
    ReconstructionRequest(
        grid=domain.grid,
        reconstruction_timestamp_utc=current_timestamp,
        measurements=sensing_service.history.records,
    )
)
```

`posterior.mean_ug_m3` and `posterior.standard_deviation_ug_m3` have the grid
shape. Active cells contain finite predictions. Inactive cells are explicitly
identified by `posterior.active_mask` and contain `NaN`, not fabricated
concentrations. Observation metadata reports the ordered fitted keys and
filtering counts. Declared noise remains authoritative on the referenced
`MeasurementRecord` values rather than being copied into a parallel metadata
array. Diagnostics report whether fitting occurred, the configured kernel,
effective numerical jitter, and log marginal likelihood.

## Computational Behavior

Exact Gaussian Process fitting costs approximately `O(n^3)` time and `O(n^2)`
memory for `n` accepted observations. Prediction is performed in deterministic
active-cell chunks to control temporary memory without approximating the
posterior. The scenario or sequential-update layer must therefore select and
document a computationally viable observation set. The current provisional R2.3
sequential profile applies the evaluated six-hour history; this
one-shot module still does not impose that policy implicitly on other callers.

`prediction_chunk_size` is an execution parameter with a default of `2048`. It
limits the number of active cells passed to each prediction call to control
temporary memory. It does not change the intended posterior and is not a
scientific or calibrated scenario parameter. Different hardware may select a
different positive chunk size.

## Limitations

- Posterior standard deviation is model uncertainty, not measured prediction error.
- The temporal kernel models statistical aging but does not encode wind-driven advection.
- The Matérn scales and prior require calibration or explicit synthetic-scenario justification.
- A configured opt-in for `CLIPPED` records treats their bounded values as Gaussian observations rather than using a censored likelihood.
- A Gaussian posterior mean can be negative; the estimator does not silently clip it and alter the fitted probabilistic model.
- Historical GP calibration excludes an entire target station, including its
  past readings and contribution to the fold's prior. Final replay uses every
  eligible observed fixed reading available at each instant.
- Complete map error and uncertainty coverage are evaluated against retained
  synthetic fields available only to the evaluator, never against a surface
  interpolated from the historical stations.
