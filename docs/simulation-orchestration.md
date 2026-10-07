# R1.3 Simulation Orchestration

## Purpose

The complete engine combines PM2.5 transport, source-qualified weather,
regional boundaries, emission sources, fixed sensors, and supplied mobile replay
positions on one UTC-aware timeline. It does not reconstruct maps, select
routes, invoke Gymnasium, or expose hidden truth through public measurements.

`iter_simulation` remains a low-level transport utility. `run_simulation` is the
R1.3 composition root used when field and measurement evidence are required.

## Time model

Every run has an aware UTC start and nonnegative elapsed seconds. Inputs queried
at elapsed time `t` govern the half-open interval `[t, next_t)`. The engine splits
transport at:

- weather transitions;
- regional-boundary transitions;
- source starts and ends;
- fixed or mobile sensor due times;
- concentration-output times;
- episode end.

Numerical stability substeps are internal to `PM25Transport.step`. Replay points
that are not measurement due times do not split transport.

`SimulationRunConfig` accepts a regular
`concentration_output_interval_s`, explicit ordered
`concentration_output_times_s`, or both. Duplicate instants from the two sources
resolve to one output boundary, and episode end remains an output. Explicit
times permit a complete field at an irregular requested instant without exposing
or persisting every internal stability substep.

After an interval reaches a boundary, the engine records a configured field,
samples due sensors in sensor-ID order, and forms a batch from numeric records.
Sensors due at `t=0` sample the initial field before interval emissions or
transport. `DROPPED` and `OUTSIDE_ACTIVE_GRID` attempts remain in history but do
not enter accepted batches; numeric `ACCEPTED` and `CLIPPED` records do.

## Replay preflight

The one-shot R1.3 engine validates the complete run before transport advances:

- the weather window has the same UTC start and duration;
- every provider exposes its transitions or is explicitly constant;
- the boundary mode and regional-boundary schedule agree;
- sensor and station references are valid;
- every mobile replay due time has an explicit position;
- source and output schedules are finite and ordered.

This is a replay contract. An interactive navigation agent cannot declare unknown
future positions and is outside the scope of R1.3. Such an agent
may later reuse the same boundary-by-boundary mechanics without changing R1.3
measurement semantics.

## Weather and lineage

Historical and explicitly synthetic windows implement the same operational
contract. Historical windows retain resource identity and hash, station, UTC
coverage, precipitation policy, and structured per-variable lineage. Synthetic
windows retain their synthetic provider and configuration and do not claim a
SENAMHI precipitation policy.

The compact `AtmosphericForcing` sent to transport contains wind and
precipitation. Structured diagnostic lineage remains separate so the engine does
not parse provenance strings or reapply upstream quality control.

## Run bundle

`write_run_bundle` writes a new destination and refuses to overwrite an existing
one. A completed bundle contains:

```text
<run-id>/
|-- manifest.json
|-- concentration.npz
|-- measurement_attempts.jsonl
|-- measurement_batches.jsonl
|-- diagnostics.jsonl
`-- checksums.json
```

`concentration.npz` stores elapsed times, UTC timestamps, the
`[time, row, column]` field, grid centers, active mask, and CRS. Attempts retain
the exact mobile positions used. Batches reference accepted attempts rather than
creating a second measurement source of truth. The manifest records the replay
input descriptor; no separate mutable used-position history is created.

`checksums.json` is written last and covers every other artifact with SHA-256.
`verify_run_bundle` identifies missing or changed files. This provides finalized,
tamper-evident evidence; it does not claim operating-system immutability or
byte-identical archives across different software platforms.

## Complete example

Run:

```bash
uv run lima-pm25-core
```

The example uses the explicitly synthetic mechanics scenario, four fixed PM2.5
sites, one supplied road-constrained mobile replay sampled every five minutes,
and seeded optional mobile noise. It writes a bundle under:

```text
resultados/r1_3_simulacion/<run-id>/
```

The console example loads `--study-config` (by default
`config/study_environment.json`) and uses its selected output schedule.
Outputs use `resultados/r1_3_simulacion/<run-id>/`.
The sensitivity experiment separately retains its summary, not a second set of
complete run bundles. Software tests are executed with pytest; they are not
another simulation product.

Repeating the same configuration reuses the already verified bundle rather than
overwriting it. The example proves mechanics and the R1.3 execution contract; its
weather, emissions, and physical coefficients remain uncalibrated and must not
be cited as observed Lima conditions.

## Comprobación funcional

El ejemplo integrado utiliza la cadencia de salida seleccionada en la
configuración: 37 campos de 0 a 10 800 s cada 300 s. Los casos de transporte,
sensores, trayectorias y orquestación comprueban las capacidades de R1.3; el total
de pruebas del repositorio incluye también otros resultados. Los casos funcionales
están en [`tests/tesis/r1_3/`](../tests/tesis/r1_3/); los contratos internos se
comprueban en [`tests/software/simulation/`](../tests/software/simulation/).
Véanse el [modelo](transport-model.md) y sus [ensayos numéricos](transport-validation.md).
