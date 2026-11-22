# Meteorología histórica como entrada del modelo

## Propósito

El adaptador de meteorología histórica conecta las series meteorológicas de
R1.1 suministradas con el modelo de transporte de PM2.5 de R1.3. No adquiere
observaciones ni realiza control de calidad, corrección o imputación. Tampoco
combina la meteorología con los registros de las estaciones de PM2.5.

La meteorología histórica permite dos usos inmediatos:

- reproducir un periodo observado seleccionado para su posterior calibración y
  evaluación;
- mantener fija la meteorología observada mientras se modifican las emisiones
  u otras entradas del escenario de PM2.5 en una ejecución contrafactual.

Las nuevas secuencias meteorológicas se generan mediante `CorrelatedWindModel`
y se integran a través de `simulation/episodes.py`; la carga del CSV no las
genera ni corrige. La [generación de episodios](episode-generation.md) explica
el ajuste y el linaje de los datos.

## Esquema del CSV suministrado

La configuración predeterminada `WeatherCsvConfig` interpreta el CSV
suministrado de la siguiente manera:

| Columna de origen | Significado | Representación de origen | Representación en la simulación |
| --- | --- | --- | --- |
| `ESTACION` | Identificador de estación | texto | una serie seleccionada explícitamente |
| `FECHA` | Fecha local | `YYYYMMDD` | marca de tiempo UTC con zona horaria |
| `HORA` | Hora local | `HH * 10000` | marca de tiempo UTC con zona horaria |
| `TEMP` | Temperatura del aire | grados Celsius | grados Celsius |
| `HR` | Humedad relativa | porcentaje | porcentaje |
| `PP` | Precipitación horaria | milímetros | `mm/h` durante un intervalo de una hora |
| `DIR_VIENTO` | Dirección de procedencia del viento | grados meteorológicos | componentes hacia el este y el norte |
| `VEL_VIENTO` | Rapidez del viento | `m/s` | componentes hacia el este y el norte |
| `<VAR>_SOURCE` | Procedencia de cada dato | vocabulario del productor | procedencia conservada |
| `PP_IFS` | Precipitación externa de ECMWF IFS | milímetros de la hora precedente | alternativa opcional explícita |

El CSV contiene los valores y la clasificación completa de procedencia de cada
dato; no tiene un archivo auxiliar asociado. Las marcas de tiempo de origen no
incluyen el desplazamiento de zona horaria. El adaptador predeterminado las
interpreta en `America/Lima` y las convierte a UTC. Los nombres de columnas, la
zona horaria, los indicadores de ausencia, el delimitador y la codificación son
campos explícitos de configuración del adaptador. Las unidades canónicas quedan
fijadas por los nombres de los campos de salida y por este contrato.

De forma predeterminada, un campo vacío, `S/D`, `NaN` o `null` se considera no
disponible. Cero es un valor válido y nunca se trata como ausente. En particular,
`PP=0` significa que no se registró precipitación para esa hora de origen.

El productor es responsable de comprobar rangos, clasificar valores atípicos,
imputar y clasificar la procedencia. El adaptador no repite esas comprobaciones.
Su política local de simulación acepta `senamhi_observed` e `imputed_*`; los
valores clasificados como atípicos operacionales y los estados ausentes siguen
sin estar disponibles. El modelo de transporte actual requiere viento y
precipitación; la temperatura y la humedad relativa se conservan como
diagnósticos opcionales. La ausencia de un valor requerido interrumpe la
cobertura continua. La política `error` rechaza la estación seleccionada ante
el primer valor requerido no disponible. El adaptador nunca realiza una nueva
imputación de los datos de origen.

`PP_IFS` es la salida de un modelo externo, no una observación de SENAMHI. No
se utiliza de forma predeterminada. Para usarla cuando `PP` de la estación no
está disponible, debe seleccionarse explícitamente
`PrecipitationSourcePolicy.STATION_THEN_IFS`.

## Procedencia

Los registros exactos de origen conservan el `SOURCE` del productor para cada
variable. La interpolación crea estados de procedencia derivados del adaptador
y registra los nombres de las fuentes originales. La precipitación constante
por tramos conserva la procedencia del extremo izquierdo. La alternativa IFS
se identifica como `ecmwf_ifs`. En condiciones de calma,
`DIR_VIENTO_SOURCE=not_applicable` es válido cuando la rapidez utilizable del
viento es cero.

`WeatherRecord` exige procedencia explícita para cada variable, incluso cuando
se construye directamente. El adaptador histórico nunca asigna una fuente
implícita `unspecified`. Los datos preparados para pruebas unitarias utilizan
su propia etiqueta explícita; la meteorología generada para escenarios utiliza
un proveedor sintético `AtmosphericForcing`, en lugar de pasar por este contrato
de registros históricos.

## Consulta temporal

`HistoricalWeatherDataset.record_at()` recibe una fecha y hora con zona horaria
y devuelve un registro meteorológico canónico en UTC. La temperatura, la
humedad relativa y las componentes del vector de viento se interpolan
linealmente entre registros adyacentes con cobertura. La dirección del viento
se reconstruye solo después de interpolar, para evitar la dirección artificial
hacia el sur que resultaría de promediar directamente 350 y 10 grados.

La precipitación no se interpola linealmente. La cantidad de origen es
numéricamente equivalente a una tasa horaria porque los registros están
separados por una hora, y el registro más reciente se mantiene constante en
`[t, t + 1h)`. El CSV suministrado no establece si la acumulación instrumental
de SENAMHI corresponde al periodo anterior o posterior a la marca de tiempo;
el intervalo indicado es una convención explícita de simulación, no una
afirmación institucional.

Una consulta anterior o posterior a la cobertura del conjunto de datos siempre
genera `WeatherCoverageError`. Nunca se interpola a través de un instante no
disponible en el producto de origen ni de un intervalo sin registros mayor que
`WeatherCsvConfig.maximum_gap`.

## Ventanas de episodios

Un episodio histórico explícito se selecciona mediante un inicio UTC con zona
horaria y una duración:

```python
from datetime import datetime, timedelta, UTC

window = weather.select_window(
    datetime(2025, 1, 1, 5, tzinfo=UTC),
    timedelta(hours=24),
)
```

El intervalo completo debe pertenecer a un único segmento de cobertura
continua. Seleccionar el mismo inicio y duración explícitos devuelve los mismos
registros. El muestreo aleatorio de episodios se controla al componer el
escenario, donde se conocen las particiones elegibles. `within_period`
restringe los registros a `[start, end)` antes de que `sample_window` elija un
inicio horario con cobertura completa. Ningún extremo de interpolación se toma
de otra partición.

`WeatherWindow` convierte los segundos transcurridos a UTC. Por tanto, puede
alimentar la ejecución actual del transporte sin modificar `PM25State`:

```python
frames = tuple(
    iter_simulation(
        model,
        initial_state,
        duration_s=window.duration_s,
        frame_duration_s=900,
        forcing_at=window.forcing_at_elapsed_s,
        forcing_change_times_s=window.forcing_change_times_s,
    )
)
```

La función que suministra la condición de frontera regional se configura de
forma independiente. Puede usar `window.utc_at_elapsed_s(elapsed_s)` cuando
necesite el mismo instante UTC.

## Límites del alcance

Actualmente, `AtmosphericForcing` envía viento y precipitación a
`PM25Transport`. La temperatura y la humedad relativa permanecen en los
registros meteorológicos canónicos para diagnósticos y futuras extensiones del
modelo, pero no alteran el transporte actual.

Este módulo no lee `pm25_lima_2025_2026_imputed.csv`. Las observaciones
históricas de PM2.5 se incorporan después a la calibración del proceso gaussiano
y a la reproducción histórica (*replay*) final mediante un adaptador separado.
La evaluación de mapas completos de R2.1/R2.2 utiliza campos sintéticos
retenidos como referencia; la meteorología sintética restringida por datos
corresponde, en cambio, a la generación de escenarios.
