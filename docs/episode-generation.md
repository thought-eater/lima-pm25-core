# Perfiles de emisión y generación de episodios

`simulation/episodes.py` compone `SimulationRunRequest` con los proveedores,
sensores y exportadores existentes. La física y trayectoria móvil proceden de
la solicitud base. Emisiones y meteorología se resuelven antes de ejecutar el
episodio. El campo completo sigue siendo una referencia sintética reservada
al simulador y al evaluador.

## Emisiones

`EmissionProfile` contiene pares `(segundos transcurridos, multiplicador)`.
Comienza en `t=0`, con tiempos estrictamente crecientes y valores finitos no
negativos. La interpolación `constant` conserva cada valor hasta el siguiente
punto, sin incluirlo; `linear` conecta los puntos. Después del último punto
conserva su valor. `starts_at_s` y `ends_at_s` delimitan la actividad de la
fuente. Los puntos son relativos al inicio del episodio, no al de la fuente.

La masa es `tasa base × integral del perfil`, incorporando longitud vial o
superficie cuando corresponde. La integral es exacta por tramos. Las
transiciones se agregan al calendario del motor, independientemente de sensores.

```python
from datetime import UTC, datetime
from lima_pm25.simulation import EmissionProfile

profile = EmissionProfile.daily(
    (
        (0, 0.5),
        (6, 0.5),
        (8, 1.5),
        (10, 1.5),
        (12, 1.0),
        (17, 1.0),
        (19, 1.4),
        (22, 0.5),
        (24, 0.5),
    ),
    start_utc=datetime(2025, 1, 1, 5, tzinfo=UTC),
    duration_s=86400,
    interpolation="linear",
)
```

`daily` interpreta horas en `America/Lima` y expande el ciclo, incluyendo
medianoche. Debe cerrar en la hora 24 con el valor de la hora 0. Los valores
del ejemplo son ilustrativos, no un perfil de tráfico medido.

Para aumentar o reducir una fuente habitual se modifica su propio perfil.
Otra fuente representa masa adicional. Esto evita contar de nuevo la emisión
habitual al representar un evento.

`AreaSource` recibe un polígono en el sistema de referencia de coordenadas (CRS)
métrico de la grilla, tasa en `µg/(m² s)`, inicio, fin y perfil. Distribuye
uniformemente por superficie de
intersección con cada celda activa. La parte exterior o inactiva **no emite
dentro del dominio ni se redistribuye**. Sin intersección positiva, la fuente
es inválida. El descriptor registra `emit_only_active_cell_intersection`.
Estos eventos se activan opcionalmente en episodios de estrés. Un incendio
hipotético es una emisión superficial equivalente: no se modelan combustión
ni ascenso de pluma.

## Meteorología

| Nombre | Valor de configuración | Construcción |
| --- | --- | --- |
| Meteorología histórica | `historical` | Ventana contigua, conservando valores y procedencia. |
| Meteorología histórica con perturbaciones sintéticas | `historical_perturbed` | Ventana histórica más perturbaciones correlacionadas del viento. |
| Meteorología sintética | `synthetic` | Ciclo horario medio más residuos correlacionados ajustados en entrenamiento. |

`CorrelatedWindModel.fit` calcula medias este/norte por hora de Lima, con al
menos dos registros por hora, y covarianza conjunta de residuos. Estima
persistencia escalar por mínimos cuadrados entre horas consecutivas cubiertas,
sin enlazar huecos. La restringe a `[0, 0.99]` para un AR(1) estacionario de
persistencia no negativa, registrando también el valor sin restricción. Es
una elección estadística, no un parámetro físico de Lima.

El proceso es `r[k] = rho*r[k-1] + sqrt(1-rho²)*epsilon[k]`. Innovaciones y
estado inicial tienen la covarianza residual estimada. Se suma el residuo al
ciclo horario o a la ventana histórica; `synthetic_variability_scale` y
`perturbation_scale` controlan las amplitudes respectivas.

Se genera una realización en nodos horarios y se interpolan componentes
entre ellos; no se promedian ángulos ni se generan ráfagas observadas
subhorarias. La dirección convencional para viento nulo es 0°.

Se limita la rapidez generada al máximo del periodo de ajuste; en la modalidad
perturbada, al mayor entre ese máximo y el de su ventana base. Se registran
límite y cantidad de nodos limitados. Esta proyección puede alterar las
estadísticas y debe examinarse al evaluar el generador. No modifica registros
históricos.

La precipitación histórica conserva secuencia, política y procedencia. La
modalidad sintética usa precipitación constante explícita (cero en el ejemplo),
no un generador aleatorio de lluvia. Los diagnósticos perturbados conservan
temperatura, humedad y precipitación de la base; el viento se identifica como
`synthetic_perturbation`, con su linaje anterior.

El ajuste representa variabilidad atmosférica horaria, no error instrumental.
El ciclo horario agrupa el periodo de ajuste: no resuelve estacionalidad ni
viento espacialmente heterogéneo. Las emisiones requieren sustento independiente.

## Particiones y reproducibilidad

`EpisodeGenerationConfig` establece periodos UTC disjuntos de entrenamiento y
evaluación, pesos meteorológicos y variaciones por fuente. Los intervalos son
`[inicio, fin)`: se filtran registros **antes** de ajustar o seleccionar
ventanas. El extremo de interpolación también pertenece a la partición.
Se elige uniformemente entre inicios horarios con cobertura completa.

El modelo usado en evaluación se ajusta solo con entrenamiento. Una ventana
y sus perturbaciones pertenecen a la misma partición. La modalidad sintética
usa la selección de ventana para obtener un inicio elegible; sus vectores
proceden del modelo ajustado, no de esa ventana.

Hay flujos aleatorios separados para modalidad, ventana, viento y sensores.
Cada fuente usa un flujo estable por identidad: añadir una fuente no cambia
las demás. Las particiones tienen espacios de semillas distintos. Las
consultas meteorológicas no consumen aleatoriedad.

La variación de fuentes configura intensidad, desplazamiento, duración y
probabilidad de activación. En perfiles diarios, el desplazamiento es de fase
local y conserva el periodo de 24 horas. En eventos transforma inicio, fin y
puntos del perfil. Un evento que no cabe se rechaza: deben configurarse rangos
compatibles con la duración.

Para un contrafactual emparejado se compone una solicitud y se cambian solo
las fuentes declaradas, conservando meteorología y semillas. Cambiar las
salidas no regenera el entorno; cambiar fronteras de integración sí puede
introducir diferencias numéricas del campo.

## Ejecución

```bash
uv run lima-pm25-core --episode-config config/episode_generation.json \
  --episode-seed 42 --episode-weather historical

uv run lima-pm25-core --episode-config config/episode_generation.json \
  --episode-seed 42 --episode-weather historical_perturbed

uv run lima-pm25-core --episode-config config/episode_generation.json \
  --episode-seed 42 --episode-weather synthetic --episode-partition evaluation
```

Sin `--episode-weather`, se selecciona según `weather_weights`.
`--episode-output` cambia el destino, inicialmente `cache/episodes/<run-id>/`.
Se reutiliza `write_run_bundle`; un destino existente se rechaza. La
configuración y las realizaciones integran la identidad de la ejecución.

La interfaz de línea de comandos usa la plantilla de tres horas del ejemplo,
sus sensores y trayectoria.
`compose_episode` acepta otras solicitudes con duración y trayectoria completas.
No amplía implícitamente trayectorias ni implementa un agente interactivo.

`config/episode_generation.json` es un ejemplo de **estrés/contrafactual**:
horarios, tasas, probabilidades y pesos son sintéticos; el polígono usa
EPSG:32718. No es un currículo seleccionado ni un inventario de emisiones.
Los episodios nominales requieren `constraint_evidence_ids`; declarar estos
identificadores no sustituye el sustento ni la evaluación de parámetros.

El ejemplo usa 2025 para ajuste y reserva 2026 para evaluación. Es una partición
de este experimento; el protocolo retenido del proceso gaussiano (GP, por sus
siglas en inglés) conserva su propia partición.
Antes de entrenar al agente deben alinearse las particiones de todos sus
insumos y reservarse las ventanas de la reproducción histórica (*replay*) final.

## Verificación

El conjunto de pruebas comprueba integrales conocidas, conservación de masa,
intersecciones de área, horarios locales, particiones, independencia aleatoria
y covarianza/persistencia en una referencia estocástica controlada.
`reconstruct_measurement_series` y las métricas existentes permiten evaluar
episodios entregando al GP solo mediciones públicas.

La diversidad reduce memorización, pero no garantiza ausencia de explotación
de la señal de recompensa (*reward hacking*) ni convierte la incertidumbre del
GP en incertidumbre meteorológica total. El replay final conserva observaciones
fijas elegibles y mediciones móviles simuladas, sin una superficie histórica
observada completa.

Referencias técnicas: [NumPy, flujos aleatorios](https://numpy.org/doc/stable/reference/random/parallel.html),
[normal multivariante](https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.multivariate_normal.html),
[Shapely, intersecciones](https://shapely.readthedocs.io/en/stable/reference/shapely.intersection.html).
