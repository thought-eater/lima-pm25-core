"""Figures and compact numeric summaries derived from reconstruction outputs."""

from datetime import timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lima_pm25.simulation import MeasurementQuality, MeasurementRecord, SensorType

from .reconstruction import (
    GaussianProcessFieldReconstructor,
    ReconstructionRequest,
    select_reconstruction_observations,
)


def used_observations(grid, config, records, timestamps, history_duration):
    """Use exactly the reconstruction eligibility rule, without fitting again."""
    return [
        select_reconstruction_observations(
            ReconstructionRequest(
                grid=grid,
                reconstruction_timestamp_utc=t,
                measurements=tuple(
                    r for r in records if t - history_duration <= r.timestamp_utc <= t
                ),
            ),
            config,
        )[0]
        for t in timestamps
    ]


def map_coverage(mean, stddev, mask):
    mean, stddev, mask = np.asarray(mean), np.asarray(stddev), np.asarray(mask)
    if mean.shape != stddev.shape or mean.ndim != 3 or mean.shape[1:] != mask.shape:
        raise ValueError("Map series and mask must align")
    m, s = mean[:, mask], stddev[:, mask]
    if not m.size:
        raise ValueError("Map series requires active cells")
    return {
        "map_count": len(mean),
        "active_cells": int(mask.sum()),
        "cell_time_count": int(m.size),
        "finite_concentration_fraction": float(np.isfinite(m).mean()),
        "valid_uncertainty_fraction": float((np.isfinite(s) & (s >= 0)).mean()),
    }


def _extent(grid):
    x0, y0, x1, y1 = grid.grid_bounds_m
    return (x0 / 1000, x1 / 1000, y0 / 1000, y1 / 1000)


def _panel(ax, field, grid, title, limits, records, timestamp, cmap="viridis"):
    image = ax.imshow(
        np.where(grid.active_mask, field, np.nan),
        origin="lower",
        extent=_extent(grid),
        vmin=limits[0],
        vmax=limits[1],
        cmap=cmap,
    )
    groups = (
        ("fixed", True, "^", "white", "Estación utilizada"),
        ("mobile", False, ".", "white", "Historial móvil utilizado"),
        ("mobile", True, "o", "red", "Medición móvil actual"),
    )
    for kind, current, marker, color, label in groups:
        selected = [
            r
            for r in records
            if r.sensor_type.value == kind
            and (kind == "fixed" or (r.timestamp_utc == timestamp) == current)
        ]
        positions = sorted({(r.x_m / 1000, r.y_m / 1000) for r in selected})
        if positions:
            x, y = zip(*positions)
            ax.scatter(
                x,
                y,
                marker=marker,
                c=color,
                s=22,
                edgecolors="black",
                linewidths=0.3,
                label=label,
            )
    ax.set(title=title, xlabel="Este UTM (km)", ylabel="Norte UTM (km)")
    ax.ticklabel_format(useOffset=False, style="plain")
    return image


def _legend(fig, axes):
    labels = {}
    for ax in np.asarray(axes).ravel():
        handles, names = ax.get_legend_handles_labels()
        labels.update(zip(names, handles))
    if labels:
        fig.legend(labels.values(), labels.keys(), loc="outside lower center", ncol=3)


def render_concentration_series(
    grid, timestamps, truth, mean, observations, indices, destination
):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    mask = grid.active_mask
    error = np.abs(mean - truth)
    # One scale across the complete series, not separately rescaled frames.
    limits = (
        float(min(truth[:, mask].min(), mean[:, mask].min())),
        float(max(truth[:, mask].max(), mean[:, mask].max())),
    )
    error_limits = (0, max(float(error[:, mask].max()), 1e-12))
    for i in indices:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
        first = _panel(
            axes[0],
            truth[i],
            grid,
            "Campo sintético de referencia",
            limits,
            observations[i],
            timestamps[i],
        )
        _panel(
            axes[1],
            mean[i],
            grid,
            "Concentración estimada",
            limits,
            observations[i],
            timestamps[i],
        )
        third = _panel(
            axes[2],
            error[i],
            grid,
            "Error absoluto",
            error_limits,
            observations[i],
            timestamps[i],
            "magma",
        )
        fig.colorbar(first, ax=axes[:2], label="PM2.5 (µg/m³)")
        fig.colorbar(third, ax=axes[2], label="Error absoluto (µg/m³)")
        elapsed = (timestamps[i] - timestamps[0]).total_seconds()
        fig.suptitle(f"t = {elapsed:g} s · {timestamps[i].isoformat()} · {grid.crs}")
        _legend(fig, axes)
        fig.savefig(destination / f"frame-{i:04d}.png", dpi=300)
        plt.close(fig)


def render_mobile_uncertainty(
    grid, timestamps, fixed_std, mixed_std, observations, index, path
):
    i = index
    reduction = fixed_std[i] ** 2 - mixed_std[i] ** 2
    mask = grid.active_mask
    maximum = max(
        float(np.nanmax(fixed_std[:, mask])), float(np.nanmax(mixed_std[:, mask]))
    )
    bound = max(float(np.max(np.abs(reduction[mask]))), 1e-12)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
    fixed_records = [r for r in observations[i] if r.sensor_type == SensorType.FIXED]
    first = _panel(
        axes[0],
        fixed_std[i],
        grid,
        "Solo estaciones fijas",
        (0, maximum),
        fixed_records,
        timestamps[i],
    )
    _panel(
        axes[1],
        mixed_std[i],
        grid,
        "Estaciones y móvil",
        (0, maximum),
        observations[i],
        timestamps[i],
    )
    last = _panel(
        axes[2],
        reduction,
        grid,
        "Reducción de varianza",
        (-bound, bound),
        observations[i],
        timestamps[i],
        "RdBu",
    )
    fig.colorbar(first, ax=axes[:2], label="Desviación estándar (µg/m³)")
    fig.colorbar(last, ax=axes[2], label="Reducción ((µg/m³)²)")
    fig.suptitle(f"Comparación con/sin historial móvil · {timestamps[i].isoformat()}")
    _legend(fig, axes)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {
        "timestamp_utc": timestamps[i].isoformat(),
        "comparison": "fixed_only_vs_fixed_plus_mobile_history",
        "integrated_reduction_ug2_m4": float(reduction[mask].sum() * grid.cell_area_m2),
        "local": [
            {
                "sensor_id": r.sensor_id,
                "variance_before": float(fixed_std[i][cell] ** 2),
                "variance_after": float(mixed_std[i][cell] ** 2),
            }
            for r in observations[i]
            if r.sensor_type == SensorType.MOBILE and r.timestamp_utc == timestamps[i]
            for cell in [grid.xy_to_cell(r.x_m, r.y_m)]
            if cell is not None and mask[cell]
        ],
    }


def render_replay(grid, timestamps, mean, stddev, observations, path, sites=()):
    indices = sorted({len(timestamps) // 2, len(timestamps) - 1})
    fig, axes = plt.subplots(
        len(indices),
        2,
        figsize=(11, 5 * len(indices)),
        squeeze=False,
        layout="constrained",
    )
    mask = grid.active_mask
    limits = (float(mean[:, mask].min()), float(mean[:, mask].max()))
    uncertainty_limits = (0, float(stddev[:, mask].max()))
    for row, i in enumerate(indices):
        first = _panel(
            axes[row, 0],
            mean[i],
            grid,
            f"Concentración · {timestamps[i].isoformat()}",
            limits,
            observations[i],
            timestamps[i],
        )
        second = _panel(
            axes[row, 1],
            stddev[i],
            grid,
            f"Incertidumbre · t = {(timestamps[i] - timestamps[0]).total_seconds():g} s",
            uncertainty_limits,
            observations[i],
            timestamps[i],
        )
        available = {
            r.sensor_id for r in observations[i] if r.sensor_type == SensorType.FIXED
        }
        for ax in axes[row]:
            for site in sites:
                if site.site_id not in available:
                    ax.scatter(
                        site.x_m / 1000,
                        site.y_m / 1000,
                        marker="x",
                        c="gray",
                        s=30,
                        label="Sitio sin observación utilizada",
                    )
                    ax.annotate(
                        site.site_id.replace("_", " ").title(),
                        (site.x_m / 1000, site.y_m / 1000),
                        fontsize=7,
                    )
    fig.colorbar(first, ax=axes[:, 0], label="PM2.5 (µg/m³)")
    fig.colorbar(second, ax=axes[:, 1], label="Desviación estándar (µg/m³)")
    fig.suptitle(f"Replay histórico · {grid.crs}")
    _legend(fig, axes)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def render_temporal_aging(grid, config, timestamp, path):
    row, column = np.argwhere(grid.active_mask)[len(np.argwhere(grid.active_mask)) // 2]
    x, y = grid.cell_center(int(row), int(column))
    measurement = MeasurementRecord(
        sensor_id="CONTROL_TEMPORAL",
        sensor_type=SensorType.FIXED,
        timestamp_utc=timestamp,
        simulation_time_s=0,
        x_m=x,
        y_m=y,
        crs=grid.crs,
        value_ug_m3=config.prior_mean_ug_m3,
        quality_status=MeasurementQuality.ACCEPTED,
        noise_stddev_ug_m3=0,
        position_source_id="controlled_cell_center",
        provenance="caso controlado de envejecimiento; medición ideal",
    )
    times = np.linspace(0, 5 * config.temporal_length_scale_s, 41)
    reconstructor = GaussianProcessFieldReconstructor(config)
    standard_deviations = [
        float(
            reconstructor.reconstruct_points(
                ReconstructionRequest(
                    grid=grid,
                    reconstruction_timestamp_utc=timestamp
                    + timedelta(seconds=float(t)),
                    measurements=(measurement,),
                ),
                ((x, y),),
            ).standard_deviation_ug_m3[0]
        )
        for t in times
    ]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    ax.plot(
        times / 3600,
        standard_deviations,
        label="Posterior: una medición, sin reemplazo",
    )
    ax.axhline(
        config.prior_stddev_ug_m3,
        linestyle="--",
        color="black",
        label="Desviación previa",
    )
    ax.set(
        xlabel="Tiempo desde la medición (h)",
        ylabel="Desviación estándar (µg/m³)",
        title="Envejecimiento de una medición en su posición exacta",
    )
    ax.legend()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return {
        "elapsed_times_s": times.tolist(),
        "standard_deviations_ug_m3": standard_deviations,
        "prior_stddev_ug_m3": config.prior_stddev_ug_m3,
        "position_m": [x, y],
        "noise_stddev_ug_m3": 0,
        "history_cutoff_applied": False,
        "temporal_length_scale_s": config.temporal_length_scale_s,
    }
