"""Ajuste causal y acotado de un GP Matérn espacio-temporal.

La media y el ruido instrumental permanecen fijados. La optimización usa
exclusivamente las mediciones del request, nunca un campo de referencia.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from time import perf_counter

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from lima_pm25.domain import GridDefinition
from lima_pm25.simulation import MeasurementBatch, MeasurementRecord, SensorType

from .reconstruction import (
    GaussianProcessFieldReconstructor,
    PosteriorField,
    ReconstructionConfig,
    ReconstructionDiagnostics,
    ReconstructionRequest,
    _fit_regressor,
    _numerical_jitter_ug_m3_squared,
    _predict_active_cells,
    select_reconstruction_observations,
)


@dataclass(frozen=True, slots=True)
class GPFitConfig:
    """Límites experimentales, no escalas atmosféricas medidas."""

    spatial_bounds_m: tuple[float, float] = (200.0, 4000.0)
    temporal_bounds_s: tuple[float, float] = (300.0, 21600.0)
    amplitude_ratio_bounds: tuple[float, float] = (0.25, 2.0)
    max_iterations: int = 20
    min_measurements: int = 12
    min_time_span_s: float = 900.0

    def __post_init__(self):
        for name in ("spatial_bounds_m", "temporal_bounds_s", "amplitude_ratio_bounds"):
            bounds = tuple(getattr(self, name))
            if (
                len(bounds) != 2
                or not all(np.isfinite(bounds))
                or not 0 < bounds[0] < bounds[1]
            ):
                raise ValueError(
                    "Los límites del ajuste deben ser positivos y ordenados"
                )
            object.__setattr__(self, name, bounds)
        if any(
            type(v) is not int or v < 1
            for v in (self.max_iterations, self.min_measurements)
        ):
            raise ValueError(
                "Iteraciones y mínimo de mediciones deben ser enteros positivos"
            )
        if not np.isfinite(self.min_time_span_s) or self.min_time_span_s < 0:
            raise ValueError("El intervalo mínimo debe ser finito y no negativo")


@dataclass(frozen=True, slots=True)
class GPParameters:
    spatial_scale_m: float
    temporal_scale_s: float
    amplitude_stddev_ug_m3: float

    def __post_init__(self):
        if any(
            not np.isfinite(v) or v <= 0
            for v in (
                self.spatial_scale_m,
                self.temporal_scale_s,
                self.amplitude_stddev_ug_m3,
            )
        ):
            raise ValueError("Los parámetros del GP deben ser positivos y finitos")

    @classmethod
    def initial(cls, config: ReconstructionConfig):
        if config.x_length_scale_m != config.y_length_scale_m:
            raise ValueError("El ajuste requiere una escala espacial isotrópica")
        return cls(
            config.x_length_scale_m,
            config.temporal_length_scale_s,
            config.prior_stddev_ug_m3,
        )

    def kernel(self):
        return ConstantKernel(self.amplitude_stddev_ug_m3**2, "fixed") * Matern(
            [self.spatial_scale_m, self.spatial_scale_m, self.temporal_scale_s],
            length_scale_bounds="fixed",
            nu=1.5,
        )


@dataclass(frozen=True, slots=True)
class GPFitDiagnostics:
    """Límites en orden amplitud/espacio/tiempo; segundos incluyen predicción."""

    attempted: bool
    converged: bool
    improved: bool
    iterations: int
    log_likelihood_before: float | None
    log_likelihood_after: float | None
    at_bounds: tuple[bool, bool, bool]
    seconds: float
    status: str


@dataclass(frozen=True, slots=True)
class AdaptivePosterior:
    posterior: PosteriorField
    parameters: GPParameters
    fit: GPFitDiagnostics


def fit_adaptive_posterior(
    request: ReconstructionRequest,
    base: ReconstructionConfig,
    settings: GPFitConfig,
    initial: GPParameters,
    *,
    prediction_chunk_size: int = 2048,
) -> AdaptivePosterior:
    """Ajustar amplitud, espacio y tiempo con gradientes de LML y L-BFGS-B.

    Las dos escalas espaciales se atan en el optimizador: su gradiente es la
    suma de los gradientes x/y. El jitter permanece anclado al perfil original.
    Se retiene el mejor punto finito evaluado, incluido el punto inicial.
    """
    started = perf_counter()
    records, metadata = select_reconstruction_observations(request, base)
    amplitude_bounds = (
        np.asarray(settings.amplitude_ratio_bounds) * base.prior_stddev_ug_m3
    )
    bounds = np.log(
        np.array(
            [
                amplitude_bounds**2,
                settings.spatial_bounds_m,
                settings.temporal_bounds_s,
            ]
        )
    )
    theta = np.log(
        [
            initial.amplitude_stddev_ug_m3**2,
            initial.spatial_scale_m,
            initial.temporal_scale_s,
        ]
    )
    if (
        not np.isfinite(theta).all()
        or np.any(theta < bounds[:, 0])
        or np.any(theta > bounds[:, 1])
    ):
        raise ValueError("La inicialización del GP debe estar dentro de los límites")
    jitter = _numerical_jitter_ug_m3_squared(base)
    model = (
        _fit_regressor(request, records, base, jitter, initial.kernel())
        if records
        else GaussianProcessRegressor(kernel=initial.kernel(), optimizer=None)
    )
    before = float(model.log_marginal_likelihood_value_) if records else None
    span = (
        (
            max(r.timestamp_utc for r in records)
            - min(r.timestamp_utc for r in records)
        ).total_seconds()
        if records
        else 0.0
    )
    positions = {(r.x_m, r.y_m) for r in records}
    attempted = (
        len(records) >= settings.min_measurements
        and span >= settings.min_time_span_s
        and len(positions) >= 3
    )
    converged, iterations, status = (
        False,
        0,
        "Datos insuficientes: se conserva la escala previa",
    )
    best_theta, best_loss = (
        theta.copy(),
        -before if before is not None else float("inf"),
    )
    if attempted:
        assert before is not None
        # Habilitar gradientes de los cuatro parámetros del kernel de sklearn.
        model.kernel_ = ConstantKernel(
            initial.amplitude_stddev_ug_m3**2, tuple(amplitude_bounds**2)
        ) * Matern(
            [
                initial.spatial_scale_m,
                initial.spatial_scale_m,
                initial.temporal_scale_s,
            ],
            length_scale_bounds=[
                settings.spatial_bounds_m,
                settings.spatial_bounds_m,
                settings.temporal_bounds_s,
            ],
            nu=1.5,
        )

        def objective(values):
            nonlocal best_theta, best_loss
            full = values[[0, 1, 1, 2]]
            likelihood, gradient = model.log_marginal_likelihood(
                full, eval_gradient=True
            )
            loss = -float(likelihood)
            jacobian = -np.array([gradient[0], gradient[1] + gradient[2], gradient[3]])
            if (
                np.isfinite(loss)
                and np.isfinite(jacobian).all()
                and loss < best_loss - 1e-8
            ):
                best_loss, best_theta = loss, values.copy()
            return loss, jacobian

        try:
            result = minimize(
                objective,
                theta,
                jac=True,
                bounds=bounds,
                method="L-BFGS-B",
                options={
                    "maxiter": settings.max_iterations,
                    "maxfun": settings.max_iterations * 5,
                    "ftol": 1e-7,
                    "gtol": 1e-5,
                },
            )
            converged, iterations = bool(result.success), int(result.nit)
            status = (
                "Convergencia"
                if converged
                else "Límite o parada del optimizador; mejor punto finito retenido"
            )
        except ValueError, FloatingPointError, np.linalg.LinAlgError:
            status = "Fallo numérico del optimizador; mejor punto finito retenido"
        parameters = (
            GPParameters(
                float(np.clip(np.exp(best_theta[1]), *settings.spatial_bounds_m)),
                float(np.clip(np.exp(best_theta[2]), *settings.temporal_bounds_s)),
                float(np.clip(np.exp(best_theta[0] / 2), *amplitude_bounds)),
            )
            if best_loss < -before - 1e-8
            else initial
        )
        model = _fit_regressor(request, records, base, jitter, parameters.kernel())
    else:
        parameters = initial
    after = float(model.log_marginal_likelihood_value_) if records else None
    mean, std = _predict_active_cells(request, model, base, prediction_chunk_size)
    posterior = PosteriorField(
        mean,
        std,
        request.grid.active_mask,
        request.reconstruction_timestamp_utc,
        metadata,
        ReconstructionDiagnostics(
            bool(records),
            str(model.kernel_ if records else model.kernel),
            jitter,
            after,
        ),
    )
    at_bounds = np.isclose(best_theta, bounds[:, 0]) | np.isclose(
        best_theta, bounds[:, 1]
    )
    fit = GPFitDiagnostics(
        attempted,
        converged,
        before is not None and after is not None and after > before + 1e-8,
        iterations,
        before,
        after,
        (bool(at_bounds[0]), bool(at_bounds[1]), bool(at_bounds[2])),
        perf_counter() - started,
        status,
    )
    return AdaptivePosterior(posterior, parameters, fit)


class AdaptiveMapUpdater:
    """Historial causal con reajuste por lote y referencia que excluye móviles nuevos.

    Las dos ramas parten del mismo parámetro previo y presupuesto de ajuste.
    Solo la rama con todo el lote se conserva. Las consultas predictivas no
    optimizan ni cambian el estado. Crear una instancia nueva por misión.
    """

    def __init__(
        self,
        grid: GridDefinition,
        base: ReconstructionConfig,
        history_duration: timedelta,
        settings: GPFitConfig,
    ):
        if history_duration.total_seconds() <= 0:
            raise ValueError("El historial debe tener duración positiva")
        self.grid, self.base, self.history_duration, self.settings = (
            grid,
            base,
            history_duration,
            settings,
        )
        self.parameters = GPParameters.initial(base)
        self.records: tuple[MeasurementRecord, ...] = ()
        self._keys: set[tuple[str, datetime]] = set()
        self.timestamp: datetime | None = None
        self.current: AdaptivePosterior | None = None

    def update(self, batch: MeasurementBatch, *, reference: bool = True):
        if self.timestamp is not None and batch.timestamp_utc <= self.timestamp:
            raise ValueError("Los lotes deben avanzar cronológicamente")
        keys = [(r.sensor_id, r.timestamp_utc) for r in batch.records]
        if len(set(keys)) != len(keys) or self._keys.intersection(keys):
            raise ValueError("Una medición no puede incorporarse dos veces")
        start = batch.timestamp_utc - self.history_duration
        previous = tuple(
            r for r in self.records if start <= r.timestamp_utc <= batch.timestamp_utc
        )
        request = ReconstructionRequest(
            grid=self.grid,
            reconstruction_timestamp_utc=batch.timestamp_utc,
            measurements=batch.records,
        )
        eligible, _ = select_reconstruction_observations(request, self.base)
        mobile = any(r.sensor_type is SensorType.MOBILE for r in eligible)
        before = None
        if mobile and reference:
            fixed = tuple(r for r in batch.records if r.sensor_type is SensorType.FIXED)
            before = fit_adaptive_posterior(
                ReconstructionRequest(
                    grid=self.grid,
                    reconstruction_timestamp_utc=batch.timestamp_utc,
                    measurements=(*previous, *fixed),
                ),
                self.base,
                self.settings,
                self.parameters,
            )
        if eligible:
            after = fit_adaptive_posterior(
                ReconstructionRequest(
                    grid=self.grid,
                    reconstruction_timestamp_utc=batch.timestamp_utc,
                    measurements=(*previous, *batch.records),
                ),
                self.base,
                self.settings,
                self.parameters,
            )
            self.parameters = after.parameters
            self.current = after
        else:
            after = None
        self.records = (*previous, *batch.records)
        self._keys.update(keys)
        self.timestamp = batch.timestamp_utc
        return before, after

    def predict(self, timestamp: datetime) -> PosteriorField:
        if self.timestamp is not None and timestamp < self.timestamp:
            raise ValueError(
                "No se puede predecir el pasado con parámetros ajustados después"
            )
        if (
            self.current is not None
            and self.current.posterior.reconstruction_timestamp_utc == timestamp
        ):
            return self.current.posterior
        records = tuple(
            r
            for r in self.records
            if timestamp - self.history_duration <= r.timestamp_utc <= timestamp
        )
        return GaussianProcessFieldReconstructor(
            self.base, kernel=self.parameters.kernel()
        ).reconstruct(
            ReconstructionRequest(
                grid=self.grid,
                reconstruction_timestamp_utc=timestamp,
                measurements=records,
            )
        )
