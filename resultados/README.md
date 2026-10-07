# Productos e indicadores verificables — R1.2–R2.3

Este índice relaciona los resultados de tesis con su configuración, procedimiento
de verificación y evidencia conservada. Los identificadores completos distinguen
las ejecuciones y permiten consultar sus manifiestos.

## Correspondencia con los IOV

| Resultado | Criterio verificable | Procedimiento y configuración | Evidencia |
| --- | --- | --- | --- |
| R1.2 | Dominio discretizado, estaciones correctamente ubicadas y justificación espacial y temporal. | [Entorno y sensibilidad](../docs/study-environment.md), [configuración](../config/study_environment.json) y [protocolo](../config/discretization_protocol.json). | [Mapa y dominio](r1_2_entorno/domain/), [informe experimental](r1_2_entorno/discretization_evidence.json) y [comprobaciones](../tests/tesis/r1_2/). |
| R1.3 | Concentración por celda e instante; posiciones móviles, error configurable y registro de mediciones y trayectorias. | [Motor integrado](../docs/simulation-orchestration.md), [modelo de transporte](../docs/transport-model.md) y [sensores](../docs/sensing.md). | [Ejemplo completo](#r13-simulación) y [pruebas funcionales y numéricas](../tests/tesis/r1_3/). |
| R2.1 | Contraste P99−P50 ≥ 20 % de la escala histórica; 100 % de mapas válidos; NMAE ≤ 0,15, NRMSE ≤ 0,20 y RMSE menor que el previo; aporte móvil por reducción de incertidumbre. | [Calibración y evaluación](../docs/historical-calibration.md) y [perfil](../config/estimation_evaluation.json). | [Mapas y métricas](#r21r22-estimación) sobre el campo sintético retenido y replay histórico. |
| R2.2 | Incertidumbre finita, no negativa y alineada; cobertura histórica de 90–98 % con exclusión completa por estación; ancho posterior menor que el previo correspondiente; respuesta a mediciones y envejecimiento. | [Proceso gaussiano](../docs/gaussian-process-reconstruction.md), [calibración](../docs/historical-calibration.md) y [casos controlados](../tests/tesis/r2_1_r2_2/). | [Calibración final y evaluación](#r21r22-estimación), mapas de incertidumbre y diagnósticos temporales. |
| R2.3 | 100 % del conjunto requerido aprobado; actualización de ambos mapas, estado cartográfico utilizable y reducción de incertidumbre por paso. | [Actualización secuencial](../docs/sequential-map-update.md), [18 casos declarados](../config/sequential_update_required_cases.json) y [pruebas](../tests/tesis/r2_3/). | [Estados, registros y resumen requerido](#r23-actualización-secuencial). |

La validación académica de la tesis corresponde a su firma. Las pruebas y
productos permiten examinar la evidencia técnica de cada resultado.

## R1.2: entorno

- [Dominio](r1_2_entorno/domain/): grafo vial, grilla, metadatos y mapa.
- [Sensibilidad espacial y temporal](r1_2_entorno/discretization_evidence.json):
  métricas de candidatos y selección de celdas de 200 m y salidas cada 300 s.

La grilla contiene 59 × 78 posiciones y 4 459 celdas activas. La selección
numérica y las limitaciones se describen en el [entorno de estudio](../docs/study-environment.md).

## R1.3: simulación

El [ejemplo `89a500ca…`](r1_3_simulacion/89a500ca2bb482ae2081accb6de5ec4ddbf6225fa7716dba3cf0d83b4ed64eec/)
conserva campos, instantes, intentos de medición, posiciones y diagnósticos:
37 campos, 53 intentos y 51 mediciones aceptadas. El campo completo es sintético;
no representa una superficie observada de Lima.

## R2.1–R2.2: estimación

- [Calibración histórica `f68cc53e…`](r2_1_r2_2_estimacion/calibracion_historica/f68cc53e2ddb33bf97540847868ae5a07563f41f016197601a1542711ac8f0a6/):
  partición, candidatos y exclusión completa de cada estación, también del previo.
- [Calibración final](r2_1_r2_2_estimacion/calibracion_historica/final_calibration.json):
  parámetros seleccionados y evaluación histórica del perfil operacional.
- [Evaluación y replay `19010dd4…`](r2_1_r2_2_estimacion/evaluacion/19010dd4072b1d5a484ca25c656265b8df72f32703b60bca99749558970726e2/):
  `synthetic_retained/` conserva el campo y las mediciones; `r2_evaluation/`,
  los mapas, métricas y figuras; `historical_replay/`, mapas y entradas con procedencia.

El perfil operacional utiliza escalas espaciales de 2 000 m, temporal de
3 600 s e historial de seis horas. La evaluación con fijas y móvil obtuvo
NMAE 0,036511, NRMSE 0,075560 y RMSE 3,131960 µg/m³, frente a
3,681459 µg/m³ del modelo basado solo en la media previa.

La cobertura histórica es 93,4028 %. Los anchos medios posterior y previo son
51,88208 y 51,88519 µg/m³: se cumple el criterio, pero la reducción es apenas
del 0,006 %. La cobertura sintética de 99,7297 % es un diagnóstico de intervalos
conservadores, fuera de la banda histórica; no debe presentarse como una mejor
calibración. El replay carece de una superficie histórica completa de referencia.

## R2.3: actualización secuencial

- [Ejemplo `8724a669…`, perfil `d4b9abf1…`](r2_3_actualizacion/example-8724a669a4989b98373cfec01b2d067d8f06938e1b4986e1e39243c0b286559c-d4b9abf19e23/):
  36 estados, lotes públicos, registros de actualización y tabla derivada.
- [Resumen de casos requeridos](r2_3_actualizacion/required-case-report-publicado.json).

El ejemplo incorpora exactamente una vez 35 mediciones móviles y conserva un
diagnóstico de reducción por lote móvil. Su paquete no contiene el campo oculto.
El resumen registra la ejecución de los 18 casos requeridos; el total del
conjunto de pruebas del repositorio no sustituye ese denominador.

## Reproducción e identidad

Los procedimientos están en [entorno](../docs/study-environment.md#reproducción),
[simulación](../docs/simulation-orchestration.md),
[calibración y evaluación](../docs/historical-calibration.md) y
[actualización secuencial](../docs/sequential-map-update.md). Ejecutarlos desde
la raíz del repositorio y usar destinos nuevos: los exportadores rechazan
sobrescribir experimentos.

Los manifiestos y hashes describen las ejecuciones conservadas, incluidas sus
rutas originales. Una reproducción usa la configuración y la identidad de su
propia ejecución. Las entradas de R1.1 se suministran en [`data/`](../data/).

Las comprobaciones de software y empaquetado se explican en
[`tests/README.md`](../tests/README.md). Las salidas temporales y los XML de
pruebas se guardan en `cache/`; las distribuciones se generan en `dist/`.
Cartografía: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright), ODbL.
