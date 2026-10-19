# Lima PM2.5 Core

Simulador espacio-temporal de PM2.5 y estimador de mapas de concentración e
incertidumbre para un área de Lima Metropolitana. Integra una red vial de
OpenStreetMap, transporte de contaminantes, sensores fijos y móviles y
reconstrucción mediante procesos gaussianos.

Este repositorio reúne el código, las configuraciones, las pruebas y los
productos que respaldan los resultados **R1.2–R2.3** de la tesis.

## Resultados

| Resultado | Contenido | Documentación |
| --- | --- | --- |
| R1.2 | Dominio georreferenciado, estaciones, escenarios y discretización espacial y temporal. | [Entorno de estudio](docs/study-environment.md) |
| R1.3 | Simulación del campo de PM2.5 y de mediciones fijas y móviles. | [Motor de simulación](docs/simulation-orchestration.md) |
| R2.1 | Reconstrucción de concentración a partir de mediciones escasas. | [Proceso gaussiano](docs/gaussian-process-reconstruction.md) |
| R2.2 | Incertidumbre posterior y evaluación de cobertura y reducción de varianza. | [Calibración y evaluación](docs/historical-calibration.md) |
| R2.3 | Actualización conjunta de concentración e incertidumbre al incorporar mediciones. | [Actualización secuencial](docs/sequential-map-update.md) |

El **[índice de productos e indicadores verificables](resultados/README.md)**
enlaza los mapas, métricas, configuraciones y ejecuciones seleccionadas.

La [generación de episodios](docs/episode-generation.md) incorpora perfiles
horarios de emisión, eventos de estrés y meteorología histórica, perturbada o
sintética correlacionada, mediante `--episode-config`.

## Alcance e interpretación

- El dominio seleccionado usa celdas horizontales de **200 m**, con 4 459 celdas
  activas. Los campos completos se conservan cada **300 s** en el ejemplo.
- El campo sintético completo es una referencia interna del simulador, accesible
  al evaluador, no una superficie observada de Lima ni una entrada del estimador.
- El replay histórico combina observaciones fijas disponibles con mediciones
  móviles simuladas, conservando su procedencia. Ceres pertenece al catálogo,
  pero no tiene valores observados de PM2.5 en el producto suministrado.
- El modelo utiliza una altura efectiva constante de 200 m como aproximación
  provisional; no resuelve perfiles verticales ni cañones urbanos.
- La planificación de rutas por aprendizaje por refuerzo y el despliegue físico
  quedan fuera del alcance de estos resultados.

## Instalación y ejecución

Se requiere [uv](https://docs.astral.sh/uv/getting-started/installation/).
Desde la raíz del repositorio, preparar Python y las dependencias fijadas en
`uv.lock`:

```bash
uv sync --locked
```

Ejecutar el ejemplo integrado de simulación:

```bash
uv run lima-pm25-core
```

La configuración seleccionada utiliza el grafo y la grilla conservados en
`resultados/r1_2_entorno/domain/`. El ejemplo guarda campos, mediciones y
diagnósticos en `resultados/r1_3_simulacion/<run-id>/` y reutiliza una ejecución
existente únicamente después de comprobar sus hashes.

Para visualizar el dominio o la demostración animada:

```bash
uv run lima-pm25-core --domain-only
uv run lima-pm25-core --animate --start-paused
```

## Organización

```text
src/lima_pm25/   dominio, simulación, estimación e interfaz de ejecución
config/         configuración del entorno y protocolos de evaluación
data/           productos horarios de entrada y su procedencia
tests/software/ pruebas de contratos internos y regresión del software
tests/tesis/    comprobaciones funcionales y numéricas por resultado
docs/           metodología, contratos y procedimientos de reproducción
resultados/     productos experimentales seleccionados
```

Los módulos de simulación y estimación consumen los contratos del dominio.
El motor genera mediciones públicas; el estimador las recibe sin acceder al
campo oculto. Todo el código pertenece a una única distribución Python.

## Pruebas y reproducción

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check --error-on-warning
uv run pytest tests/software
uv run pytest tests/tesis
uv run pytest
uv build
```

La [guía de pruebas](tests/README.md) explica qué verifica cada conjunto.
Las pruebas automatizadas no sustituyen los experimentos de discretización,
calibración y evaluación: sus procedimientos y productos se encuentran en el
[índice de resultados](resultados/README.md).

Los [datos de entrada](data/README.md) conservan valores, ausencias y procedencia
por variable. Cartografía: © [OpenStreetMap contributors](https://www.openstreetmap.org/copyright),
licencia ODbL.
