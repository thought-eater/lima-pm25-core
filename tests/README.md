# Pruebas y verificación de resultados

La organización distingue el propósito de una comprobación de la herramienta
que la ejecuta. Ambos conjuntos utilizan pytest y cada caso tiene una sola
implementación.

## Pruebas de software

[`software/`](software/) contiene comprobaciones de contratos internos,
validación de entradas, lectura y escritura, cálculo de métricas, estabilidad,
aislamiento de datos, exportadores, interfaz de ejecución y arquitectura.
Incluye pruebas unitarias y de integración de componentes.

Por ejemplo, `software/estimation/test_evaluation.py` verifica el cálculo de
MAE, RMSE y cobertura con arreglos controlados. Que pase no demuestra que el
modelo alcance los umbrales en los datos del estudio.

```bash
uv run pytest tests/software
```

## Comprobaciones de tesis

[`tesis/`](tesis/) agrupa las pruebas funcionales y numéricas por resultado:

| Directorio | Propósito | Alcance de la evidencia |
| --- | --- | --- |
| [r1_2/](tesis/r1_2/) | Ubicación de estaciones, protocolo y lectura del informe de discretización. | Comprueba configuración y productos conservados; no repite la sensibilidad. |
| [r1_3/](tesis/r1_3/) | Transporte frente a referencias numéricas, campos por instante, mediciones, errores, posiciones y registros. | Verificación funcional y numérica del simulador. |
| [r2_1_r2_2/](tesis/r2_1_r2_2/) | Correspondencia del perfil con la calibración seleccionada, aislamiento del campo y respuesta de incertidumbre. | Comprobaciones controladas que complementan la evaluación experimental. |
| [r2_3/](tesis/r2_3/) | Actualización secuencial e integración con R1.3. | Los 18 casos requeridos: 16 funcionales y dos de integración. |

Los conjuntos funcionales incluyen los casos inválidos y límites necesarios
para verificar cada capacidad. Los casos de R2.3 se identifican explícitamente
en [`sequential_update_required_cases.json`](../config/sequential_update_required_cases.json).
Su denominador no es el total de pruebas del repositorio.

```bash
uv run pytest tests/tesis
uv run pytest tests/tesis/r2_3 --junitxml=cache/tests/required-cases.xml
```

El XML es una salida temporal para calcular el resumen de casos requeridos.
El producto de tesis es el [resumen seleccionado](../resultados/README.md#r23-actualización-secuencial)
junto con los estados y registros del ejemplo.

## Experimentos y productos

Los experimentos se ejecutan mediante las funciones y opciones del programa:

- R1.2: `--study-sensitivity`, según el
  [protocolo de discretización](../docs/study-environment.md#reproducción).
- R1.3: ejemplo integrado con `uv run lima-pm25-core`.
- Episodios con perfiles y meteorología variable: `--episode-config`, según la
  [guía de generación](../docs/episode-generation.md). Sus ejecuciones
  exploratorias se guardan inicialmente en `cache/episodes/`.
- R2.1–R2.2: `--historical-calibration` y `--estimation-evaluation`, según el
  [procedimiento de calibración y evaluación](../docs/historical-calibration.md).
- R2.3: procesamiento secuencial de lotes y resumen del conjunto requerido,
  según la [guía de actualización](../docs/sequential-map-update.md).

El [índice de productos](../resultados/README.md) relaciona IOV, configuraciones,
procedimientos y evidencia. Los campos sintéticos completos se reservan al
evaluador; los ensayos no establecen exactitud atmosférica real.

## Verificación completa

Ejecutar desde la raíz del repositorio:

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check --error-on-warning
uv run pytest
uv build
```

La configuración de pytest descubre ambos conjuntos. Las pruebas usan entradas
locales o sintéticas y no solicitan descargas de Overpass. La construcción
comprueba el empaquetado, no constituye un experimento científico.
