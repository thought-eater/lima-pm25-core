# PM2.5 Sensing

## Scientific Role

The simulator supports two explicit scenario families:

- Contract-qualified historical/replay scenarios preserve supplied SENAMHI
  values, timestamps, positions, missingness, and provenance.
- Synthetic scenarios generate a hidden PM2.5 field and sample it at finite
  sensor positions. Their inputs must be calibrated or constrained by accepted
  Lima evidence.

Sparse historical station observations are not a complete ground-truth field.
Generated or imputed values must not be relabeled as observed ground truth.
Conditions outside Lima's plausible range require an explicit synthetic stress
test or counterfactual label.

The sensing service samples the simulated hidden field.
Historical PM2.5 time-series ingestion is a separate contract and
must not use the historical-weather adapter.

## Fixed Sites

`lima_pm25.domain` owns immutable PM2.5 station locations. The initial catalog
contains the four SENAMHI PM2.5 sites inside the study rectangle:

| Site ID | Longitude | Latitude | Altitude (m) |
| --- | ---: | ---: | ---: |
| `CAMPO_DE_MARTE` | -77.0432 | -12.0705 | 117 |
| `CERES` | -76.92706 | -12.02869 | unavailable |
| `SANTA_ANITA` | -76.9714 | -12.043 | 253 |
| `SAN_JUAN_DE_LURIGANCHO` | -76.99925 | -11.9808 | 240 |

These coordinates come from `pm25_lima_2025_2026_imputed.csv` under contract
`senamhi-hourly-imputed` version `1.0.0`. They are projected from WGS84 into the
domain CRS with longitude/latitude axis order and must fall inside the projected
boundary and an active grid cell.

Meteorological-product coordinates remain tied to the meteorological resource.
Similar station names do not justify averaging or substituting coordinates
between products.

## Sensor Configuration

A sensor configuration describes measurement behavior without owning a fixed
site or mobile trajectory. Fixed sensors reference a station-site ID. Mobile
sensors reference a provider that declares its identity and projected CRS and
returns an explicit `(x_m, y_m)` for a sensor ID and elapsed simulation time.
The sensing service does not interpolate, extrapolate, or repair a missing
position.

The initial fixed configuration is hourly and starts at `t=0`, assuming that the
episode starts on an exact clock hour. Mobile interval and first due time are
both explicit per sensor. Five minutes (`300 s`) is provisional scenario input,
not a calibrated scientific default.

Every runtime state retains its next due elapsed time. Advancing across several
due instants processes each one in chronological order, with simultaneous ties
ordered by sensor ID.

Sensor cadence does not control source emissions. Synthetic point and road
sources emit continuous interval mass from rates in `ug/s` or `ug/(m s)` and may
start, stop, or change at arbitrary simulation seconds. Source transitions,
weather changes, sensor due times, and configured outputs are independent engine
boundaries. Solver-stability substeps remain internal to transport; supplied
trajectory points affect execution only when a sensor position is due.

A mobile position may follow a permitted road briefly outside the concentration
grid. That due time produces an `OUTSIDE_ACTIVE_GRID` attempt containing its
position but no numeric PM2.5 value. The attempt remains in measurement history,
is excluded from reconstruction inputs, and the sensor advances to its next
configured due time.

## Truth Sampling

Concentration uses `ug/m3`. Positions and grid distances use meters in the
projected domain CRS. The transport model stores one finite-volume concentration
per active control cell. Sensor sampling therefore:

1. Resolves the projected position with `GridDefinition.xy_to_cell()`.
2. Returns that cell's concentration when it is active.
3. Raises `SamplingSupportError` outside the grid or in an inactive cell when
   called directly.

The sampler never interpolates neighbors or repairs an unsupported position by
searching for a nearest active cell. Internal cell faces use the domain's
half-open convention: a point exactly on a face belongs to the cell to its east
or north. Mobile readings can jump when they cross a face because the finite-
volume state contains no resolved sub-cell gradient.
`SensingService` calls this sampler only when a mobile position belongs to an
active cell; otherwise it records a failed `OUTSIDE_ACTIVE_GRID` attempt and
continues.

## Observation Errors

The observation pipeline is:

```text
candidate = sampled_truth + bias + gaussian_noise
dropout decision
fixed zero lower bound and optional configured upper saturation
```

The Lima baseline uses zero bias, zero noise standard deviation, and zero
dropout. Its accepted observation therefore equals the containing cell's hidden
truth exactly. Error behavior is activated only through explicit sensor
configuration.

A dropped attempt has quality `DROPPED` and `value_ug_m3=None`; it is not a
zero-concentration observation. A numeric value changed by the zero lower bound
or optional upper saturation has quality `CLIPPED`. No generic upper value is
assumed: saturation applies only when a scenario configures it explicitly. An
unsupported mobile position has quality
`OUTSIDE_ACTIVE_GRID` and also has no numeric value, but remains distinguishable
from random dropout. Seeded stochastic behavior uses only the episode NumPy
random generator and deterministic `(timestamp, sensor_id)` processing order.

## Records And Isolation

Public `MeasurementRecord` values contain sensor identity and type, aware UTC
time, elapsed simulation time, event-time projected position and CRS, observed
value, quality, declared noise, position-source identity, and provenance. Mobile
records use their position provider ID as `position_source_id`; the applicable
mobility restrictions belong to the run configuration rather than each attempt.
Records do not contain sampled hidden truth or any pre-error value.

The full history retains attempts in deterministic chronological order.
`accepted_records` returns only records with finite numeric observations. The
sensing service does not retain a second private history of sampled truth,
random draws, or pre-bound values; reproducibility relies on the recorded
configuration, seed, software environment, fields, positions, and public
attempts.

Emission sources and sensors are unrelated entities: sources add PM2.5 mass to
the hidden field, while sensors observe the resulting concentration. Neither
sensor state nor measurements are stored on the OSMnx road graph.
