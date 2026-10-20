# Simulación de mediciones de PM2.5

## Función en los resultados de tesis

El simulador admite dos familias explícitas de escenarios:

- Los escenarios históricos o de reproducción histórica (*replay*) conservan
  los valores, marcas de tiempo, posiciones, ausencias y procedencia de SENAMHI
  según el contrato del producto suministrado.
- Los escenarios sintéticos generan un campo oculto de PM2.5 y lo muestrean en
  las posiciones de un conjunto finito de sensores. Sus entradas deben estar
  calibradas o restringidas por evidencia aceptada para Lima.

Las observaciones históricas dispersas de estaciones no constituyen un campo
completo de referencia. Los valores generados o imputados no deben presentarse
como valores reales observados. Las condiciones fuera del rango plausible de
Lima deben identificarse explícitamente como un escenario sintético de estrés
o un contrafactual.

El servicio de medición muestrea el campo oculto simulado. La incorporación de
series históricas de PM2.5 tiene un contrato separado y no debe usar el
adaptador de meteorología histórica.

## Estaciones fijas

`lima_pm25.domain` mantiene las ubicaciones inmutables de las estaciones de
PM2.5. El catálogo inicial contiene los cuatro sitios de SENAMHI dentro del
rectángulo de estudio:

| Identificador del sitio | Longitud | Latitud | Altitud (m) |
| --- | ---: | ---: | ---: |
| `CAMPO_DE_MARTE` | -77.0432 | -12.0705 | 117 |
| `CERES` | -76.92706 | -12.02869 | no disponible |
| `SANTA_ANITA` | -76.9714 | -12.043 | 253 |
| `SAN_JUAN_DE_LURIGANCHO` | -76.99925 | -11.9808 | 240 |

Estas coordenadas proceden de `pm25_lima_2025_2026_imputed.csv`, bajo el
contrato `senamhi-hourly-imputed`, versión `1.0.0`. Se proyectan desde WGS84 al
sistema de referencia de coordenadas (CRS) del dominio con orden de ejes
longitud/latitud y deben quedar dentro del límite proyectado y de una celda
activa de la grilla.

Las coordenadas del producto meteorológico permanecen ligadas a su propio
recurso. La similitud de nombres entre estaciones no justifica promediar ni
sustituir coordenadas entre productos.

## Configuración de sensores

La configuración de un sensor describe su comportamiento de medición sin
contener la ubicación fija ni la trayectoria móvil. Los sensores fijos
referencian un identificador de sitio del catálogo. Los móviles referencian un
proveedor que declara su identidad y su CRS proyectado, y devuelve una posición
explícita `(x_m, y_m)` para un identificador de sensor y un tiempo transcurrido
de simulación. El servicio de medición no interpola, extrapola ni corrige una
posición ausente.

La configuración fija inicial es horaria y comienza en `t=0`, suponiendo que el
episodio empieza en una hora exacta del reloj. El intervalo móvil y el primer
instante programado son explícitos para cada sensor. Cinco minutos (`300 s`)
es una entrada provisional del escenario, no un valor científico predeterminado
y calibrado.

Cada estado de ejecución conserva el próximo tiempo transcurrido de medición.
Si se avanza a través de varios instantes programados, se procesa cada uno en
orden cronológico; las mediciones simultáneas se ordenan por identificador de
sensor.

La cadencia de los sensores no controla las emisiones. Las fuentes sintéticas
puntuales y viales emiten masa continua por intervalo a partir de tasas en
`ug/s` o `ug/(m s)`, y pueden comenzar, terminar o cambiar en segundos
arbitrarios de la simulación. Las transiciones de fuentes, los cambios
meteorológicos, los instantes de medición y las salidas configuradas delimitan
intervalos del motor de forma independiente. Los subpasos de estabilidad del
integrador numérico son internos al transporte; los puntos de trayectoria
suministrados afectan la ejecución solo cuando corresponde obtener la posición
de un sensor para medir.

Una posición móvil puede seguir una vía permitida brevemente fuera de la grilla
de concentración. En ese instante se genera un intento `OUTSIDE_ACTIVE_GRID`
con su posición, pero sin valor numérico de PM2.5. El intento permanece en el
historial de mediciones, se excluye de las entradas de reconstrucción y el
sensor avanza a su próximo instante programado.

## Muestreo del campo oculto

La concentración se expresa en `ug/m3`. Las posiciones y distancias de la grilla
se expresan en metros del CRS proyectado del dominio. El modelo de transporte
almacena una concentración de volumen finito por celda de control activa. Por
ello, el muestreo del sensor:

1. Localiza la posición proyectada con `GridDefinition.xy_to_cell()`.
2. Devuelve la concentración de esa celda si está activa.
3. Genera `SamplingSupportError` fuera de la grilla o en una celda inactiva
   cuando se invoca directamente.

El muestreador nunca interpola entre celdas vecinas ni corrige una posición sin
soporte buscando la celda activa más cercana. Las caras internas usan la
convención semiabierta del dominio: un punto exactamente sobre una cara
pertenece a la celda situada al este o al norte. Las lecturas móviles pueden
presentar saltos al cruzar una cara porque el estado de volúmenes finitos no
resuelve gradientes dentro de una celda.
`SensingService` invoca este muestreador solo cuando la posición móvil pertenece
a una celda activa; en caso contrario, registra un intento fallido
`OUTSIDE_ACTIVE_GRID` y continúa.

## Errores de medición

La secuencia de medición es:

```text
candidate = sampled_truth + bias + gaussian_noise
decisión de pérdida de medición
límite inferior fijo en cero y saturación superior opcional configurada
```

En el esquema anterior, `sampled_truth` es la concentración interna muestreada,
`bias` es el sesgo y `gaussian_noise` es el ruido gaussiano. Después se decide
si se pierde la medición (*dropout*) y se aplica un límite inferior fijo de
cero y, si está configurada, una saturación superior.

La configuración de referencia para Lima utiliza sesgo nulo, desviación
estándar del ruido nula y probabilidad de pérdida nula. Por tanto, su medición
aceptada coincide exactamente con el campo oculto de la celda que la contiene.
Los errores solo se activan mediante configuración explícita del sensor.

Un intento perdido tiene calidad `DROPPED` y `value_ug_m3=None`; no es una
medición de concentración cero. Un valor numérico modificado por el límite
inferior cero o por la saturación superior opcional tiene calidad `CLIPPED`.
No se supone un límite superior genérico: la saturación se aplica solo cuando
el escenario la configura explícitamente. Una posición móvil sin soporte tiene
calidad `OUTSIDE_ACTIVE_GRID` y tampoco contiene valor numérico, pero se
distingue de una pérdida aleatoria. El comportamiento estocástico con semilla
utiliza únicamente el generador aleatorio de NumPy del episodio y el orden de
procesamiento determinista `(timestamp, sensor_id)`.

## Registros y aislamiento del campo oculto

Los registros públicos `MeasurementRecord` contienen identidad y tipo de
sensor, tiempo UTC con zona horaria, tiempo transcurrido de simulación,
posición proyectada en el instante de medición y su CRS, valor medido, calidad,
ruido declarado, identidad de la fuente de posición y procedencia. Los
registros móviles utilizan el identificador del proveedor de posiciones como
`position_source_id`; las restricciones de movilidad aplicables pertenecen a
la configuración de la ejecución, no a cada intento. Los registros no contienen
el valor muestreado del campo oculto ni valores anteriores a los errores.

El historial completo conserva los intentos en orden cronológico determinista.
`accepted_records` devuelve solo registros con mediciones numéricas finitas.
El servicio de medición no conserva un segundo historial privado del campo
muestreado, de los sorteos aleatorios ni de los valores anteriores a la
aplicación de límites. La reproducibilidad depende de la configuración, la
semilla, el entorno de software, los campos, las posiciones y los intentos
públicos registrados.

Las fuentes de emisión y los sensores son entidades independientes: las
fuentes añaden masa de PM2.5 al campo oculto y los sensores miden la
concentración resultante. Ni el estado de los sensores ni las mediciones se
almacenan en el grafo vial de OSMnx.
