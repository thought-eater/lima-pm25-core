# Actualización secuencial de mapas — R2.3

## Función y configuración

`lima_pm25.estimation.SequentialMapUpdater` consume los lotes públicos
`MeasurementBatch` de R1.3 y publica estados cartográficos inmutables con
concentración e incertidumbre del mismo posterior. La simulación produce las
mediciones sin invocar al estimador.

El [perfil seleccionado](../config/estimation_evaluation.json) utiliza escalas
espaciales de 2 000 m, escala temporal de 3 600 s e historial inclusivo de seis
horas. La escala espacial del proceso gaussiano (GP, por sus siglas en inglés)
es distinta del lado de celda de 200 m.
Los registros `CLIPPED` se excluyen. Los parámetros y sus limitaciones se
describen en la [calibración histórica](historical-calibration.md).

## Semántica de actualización

Un lote reúne todas las mediciones numéricas públicas de un instante UTC.
El módulo incorpora sus registros elegibles antes de publicar un único estado:
dos mediciones móviles simultáneas generan una versión del mapa, con ambas
mediciones contabilizadas exactamente una vez.

- Los instantes procesados deben ser estrictamente crecientes.
- Se rechazan duplicados, lotes antiguos y adiciones tardías al mismo instante.
- Un error de validación o reconstrucción conserva el estado previo.
- Un lote compuesto solo por registros excluidos se registra sin crear un mapa.
- Para el instante `t` se utiliza el historial en `[t − 6 h, t]`. Las claves de
  registros que salen de esa ventana se conservan para detectar duplicados.

El límite de seis horas es una política de historial evaluada, no una afirmación
de que la información atmosférica desaparezca físicamente después de ese plazo.

## Estado publicado y reducción de incertidumbre

Cada `SequentialMapSnapshot` contiene número de secuencia, instante de
reconstrucción, corte observacional, grilla, máscara activa, media, desviación
estándar y diagnósticos. Es el componente cartográfico que puede consumir un
entorno de aprendizaje; no incluye acciones, recompensas o el ciclo de un agente.
El campo oculto y los valores anteriores al error instrumental no se exponen.

Para cada lote aplicado se comparan, **en el mismo instante de predicción**:

```text
antes: historial previamente incorporado
después: historial previo más el lote nuevo completo
reducción: varianza antes − varianza después
```

Así se distingue ganancia de información de envejecimiento temporal. El primer
estado «antes» es el previo configurado. Se registran varianza local en las
celdas observadas e integrada sobre el dominio activo. Sus unidades son
`(µg/m³)²` y `(µg/m³)² m²`, respectivamente.

Las mediciones redundantes pueden producir reducción cercana a cero. Los lotes
móviles deben mantener la varianza no creciente dentro de tolerancia; el caso
controlado informativo exige reducción positiva.

## Ejemplo reproducible

Ejecutar este fragmento Python desde la raíz con un destino nuevo:

```python
from pathlib import Path

from lima_pm25.domain import (
    create_lima_pm25_station_catalog,
    create_spatial_domain,
    load_study_environment_config,
)
from lima_pm25.estimation import (
    SequentialMapUpdater,
    SequentialUpdateConfig,
    load_change10_evaluation_profile,
    process_measurement_batches,
    verify_sequential_update_bundle,
    write_sequential_update_bundle,
)
from lima_pm25.simulation import run_complete_synthetic_demo

study = load_study_environment_config("config/study_environment.json")
domain = create_spatial_domain(study.domain)
stations = create_lima_pm25_station_catalog(domain)
run, _ = run_complete_synthetic_demo(
    domain,
    output_directory=Path("cache/reproduccion/r13"),
    station_catalog=stations,
    study_config=study,
)
profile = load_change10_evaluation_profile("config/estimation_evaluation.json")
config = SequentialUpdateConfig.from_change10_profile(profile)
trace = process_measurement_batches(
    SequentialMapUpdater(domain.grid, config),
    run.measurement_batches,
)
destination = Path("cache/reproduccion/r23")
write_sequential_update_bundle(
    trace=trace,
    batches=run.measurement_batches,
    config=config,
    grid=domain.grid,
    destination=destination,
    upstream_manifest_json=run.manifest_json,
)
assert verify_sequential_update_bundle(destination) == ()
```

El paquete de R2.3 conserva lotes públicos, matrices de estados, registros de
actualización, configuración, identidad de la ejecución de entrada y hashes.
El paquete de R1.3 puede contener el campo sintético para evaluación; el de
R2.3 conserva únicamente entradas públicas y salidas del estimador.

`update_summary.csv` deriva de los lotes y registros existentes. Sus cantidades
integradas se refieren al **lote completo** y se repiten por medición: no deben
sumarse entre filas de un mismo lote ni atribuirse a una sola medición simultánea.

## Verificación del IOV

El [manifiesto requerido](../config/sequential_update_required_cases.json)
establece 18 casos: 16 funcionales y dos de integración con R1.3.

```bash
uv run pytest tests/tesis/r2_3 --junitxml=cache/tests/required-cases.xml
```

El exportador `write_r23_required_case_report` compara ese XML con el manifiesto
y el `verification_report.json` del ejemplo. Se exige que todos los casos estén
presentes y aprobados, sin fallos, errores, omisiones ni ausencias. La evidencia
de ejecución comprueba además:

- incorporación exacta del 100 % de las mediciones móviles elegibles;
- diagnóstico de incertidumbre para cada lote móvil aplicado;
- ausencia de incremento local o integrado fuera de tolerancia;
- reducción positiva en al menos un lote informativo del ejemplo.

Para producir un resumen a partir del ejemplo conservado:

```python
from pathlib import Path
from lima_pm25.estimation import (
    load_change10_evaluation_profile,
    write_r23_required_case_report,
)

profile = load_change10_evaluation_profile("config/estimation_evaluation.json")
bundle = Path(
    "resultados/r2_3_actualizacion/"
    "example-8724a669a4989b98373cfec01b2d067d8f06938e1b4986e1e39243c0b286559c-d4b9abf19e23"
)
write_r23_required_case_report(
    required_case_manifest="config/sequential_update_required_cases.json",
    junit_xml="cache/tests/required-cases.xml",
    runtime_verification_report=bundle / "verification_report.json",
    destination="cache/reproduccion/r23-required-case-report.json",
    configuration_sha256=profile.source_sha256,
)
```

El [ejemplo seleccionado](../resultados/README.md#r23-actualización-secuencial)
conserva 36 estados y 35 mediciones móviles incorporadas exactamente una vez.
La comprobación funcional no establece latencia de tiempo real ni desempeño de
campo. El conjunto completo de software y la construcción del paquete se
ejecutan por separado de este criterio de aceptación.
