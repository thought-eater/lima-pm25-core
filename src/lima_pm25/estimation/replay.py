from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime, timedelta
from itertools import pairwise

from lima_pm25.domain import GridDefinition
from lima_pm25.simulation import MeasurementBatch, MeasurementRecord, SensorType

from .reconstruction import (
    GaussianProcessFieldReconstructor,
    PosteriorField,
    ReconstructionConfig,
    ReconstructionRequest,
)


def compose_historical_replay_batches(
    *,
    historical_fixed_records: Sequence[MeasurementRecord],
    simulated_mobile_attempts: Sequence[MeasurementRecord],
) -> tuple[MeasurementBatch, ...]:
    fixed = tuple(historical_fixed_records)
    mobile = tuple(simulated_mobile_attempts)
    if any(
        record.sensor_type is not SensorType.FIXED or not record.is_accepted
        for record in fixed
    ):
        raise ValueError(
            "historical_fixed_records must contain accepted fixed observations"
        )
    if any(record.sensor_type is not SensorType.MOBILE for record in mobile):
        raise ValueError("simulated_mobile_attempts must contain only mobile records")

    accepted_mobile = tuple(record for record in mobile if record.is_accepted)
    grouped: dict[datetime, list[MeasurementRecord]] = defaultdict(list)
    keys: set[tuple[str, datetime]] = set()
    for record in (*fixed, *accepted_mobile):
        key = (record.sensor_id, record.timestamp_utc)
        if key in keys:
            raise ValueError(
                f"Duplicate replay measurement for sensor {record.sensor_id!r} "
                f"at {record.timestamp_utc.isoformat()}"
            )
        keys.add(key)
        grouped[record.timestamp_utc].append(record)

    batches = []
    for timestamp in sorted(grouped):
        records = tuple(sorted(grouped[timestamp], key=lambda record: record.sensor_id))
        simulation_times = {record.simulation_time_s for record in records}
        if len(simulation_times) != 1:
            raise ValueError(
                "Replay records sharing a timestamp must share simulation_time_s"
            )
        batches.append(
            MeasurementBatch(
                simulation_time_s=records[0].simulation_time_s,
                timestamp_utc=timestamp,
                records=records,
            )
        )
    return tuple(batches)


def reconstruct_measurement_series(
    *,
    grid: GridDefinition,
    config: ReconstructionConfig,
    batches: Sequence[MeasurementBatch],
    reconstruction_timestamps_utc: Sequence[datetime],
    history_duration: timedelta,
    prediction_chunk_size: int = 2_048,
) -> tuple[PosteriorField, ...]:
    if history_duration <= timedelta(0):
        raise ValueError("history_duration must be positive")
    timestamps = tuple(reconstruction_timestamps_utc)
    if any(current <= previous for previous, current in pairwise(timestamps)):
        raise ValueError("reconstruction_timestamps_utc must strictly increase")
    records = tuple(record for batch in batches for record in batch.records)
    reconstructor = GaussianProcessFieldReconstructor(
        config,
        prediction_chunk_size=prediction_chunk_size,
    )
    posteriors = []
    for timestamp in timestamps:
        if timestamp.tzinfo is None or timestamp.utcoffset() != timedelta(0):
            raise ValueError("reconstruction timestamps must use UTC")
        history_start = timestamp - history_duration
        selected = tuple(
            record
            for record in records
            if history_start <= record.timestamp_utc <= timestamp
        )
        posteriors.append(
            reconstructor.reconstruct(
                ReconstructionRequest(
                    grid=grid,
                    reconstruction_timestamp_utc=timestamp,
                    measurements=selected,
                )
            )
        )
    return tuple(posteriors)
