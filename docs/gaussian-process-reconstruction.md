# Reconstrucción mediante procesos gaussianos

## Función en los resultados de tesis

`lima_pm25.estimation` reconstruye el campo actual de PM2.5 a partir de
mediciones dispersas de sensores. La media posterior es la estimación de
concentración de R2.1 y la
desviación estándar posterior es la incertidumbre del modelo asociada a R2.2.
Solo hay mediciones en las posiciones de los sensores; los valores en otras
posiciones son estimaciones y no deben presentarse como valores reales observados.

El estimador recibe registros públicos `MeasurementRecord` con la concentración medida,
la posición en el instante de medición, la marca de tiempo UTC, la desviación
estándar declarada del ruido aleatorio, la calidad y la procedencia. La solicitud
de reconstrucción no contiene el campo oculto del simulador ni los valores
anteriores a la aplicación de errores de medición.

## Modelo

La función de covarianza (*kernel*) del proceso gaussiano (GP, por sus siglas en
inglés) es:

```text
ConstantKernel(prior_variance)
    * Matern(length_scale=[x_m, y_m, time_s], nu=1.5)
```

Las coordenadas se centran respecto de los límites suroeste de la grilla, pero
se mantienen en metros del sistema proyectado. El tiempo se expresa en segundos
relativos al instante de reconstrucción solicitado: las celdas donde se predice
usan tiempo cero y las observaciones pasadas usan tiempos relativos negativos.
Los residuos de concentración se ajustan respecto de la media previa constante
configurada. Por ello, una predicción alejada de las observaciones se aproxima
a la media y a la desviación estándar previas.

Todas las escalas de longitud son explícitas y permanecen fijas durante una
reconstrucción. Su selección y sus limitaciones se documentan en la
[calibración histórica](historical-calibration.md). El perfil operacional se
selecciona con los datos de calibración antes de evaluar los campos sintéticos
retenidos.

## Envejecimiento temporal de la información

Una observación conserva el valor registrado en su instante original. Su
capacidad de describir un campo posterior disminuye al aumentar la diferencia
entre su marca de tiempo y el instante de reconstrucción, en relación con
`temporal_length_scale_s`. Por tanto, la incertidumbre es prácticamente cero en
la posición y el instante exactos de una observación sin ruido, y crece hacia la
incertidumbre previa en instantes posteriores.

Se utiliza cada observación aceptada incluida en `ReconstructionRequest` cuyo
instante sea anterior o igual al de reconstrucción. Quien invoca al módulo
puede proporcionar un subconjunto determinista adecuado al escenario; el
reconstructor no impone otra ventana temporal implícita, un máximo de muestras
ni una eliminación por antigüedad. La relevancia temporal dentro del conjunto
recibido se representa mediante la función de covarianza.

## Ruido de las mediciones

Los sensores simulados de referencia declaran sesgo y ruido aleatorio nulos.
Para cada observación aceptada, el regresor recibe:

```text
observation_variance = noise_stddev_ug_m3 ** 2
numerical_jitter = 1e-10 * prior_stddev_ug_m3 ** 2
alpha = observation_variance + numerical_jitter
```

El término de estabilización numérica (*jitter*), relativo a la escala del
modelo, solo estabiliza la matriz de covarianza y se registra por separado del
ruido físico del sensor. Es una salvaguarda numérica interna, no un parámetro
del escenario. No se utiliza `WhiteKernel`: un nivel mínimo global de ruido
residual impediría que la incertidumbre posterior se aproximara a cero ante una
medición exacta del sensor de referencia. Si se configura explícitamente ruido
aleatorio del sensor, este se representa mediante el `alpha` de cada
observación; el sesgo sistemático no se reinterpreta como varianza.
El factor relativo `1e-10` sigue la escala de la diagonal de la matriz de
covarianza y es lo bastante pequeño para no convertir, de manera significativa,
una observación declarada sin ruido en una observación ruidosa.

Los registros numéricos `CLIPPED` se excluyen de forma predeterminada porque un
límite reportado no es necesariamente una concentración exacta. Configurar
`include_clipped_observations=True` permite tratar esos límites como
aproximaciones numéricas ordinarias; no implementa una función de verosimilitud
para observaciones censuradas. Los metadatos contabilizan por separado los
registros recortados excluidos por esta política y los intentos con pérdida de
medición.

## Uso

```python
from lima_pm25.estimation import (
    GaussianProcessFieldReconstructor,
    ReconstructionConfig,
    ReconstructionRequest,
)

config = ReconstructionConfig(
    prior_mean_ug_m3=20.0,
    prior_stddev_ug_m3=15.0,
    x_length_scale_m=1_000.0,
    y_length_scale_m=1_000.0,
    temporal_length_scale_s=3_600.0,
    include_clipped_observations=False,
)
reconstructor = GaussianProcessFieldReconstructor(
    config,
    prediction_chunk_size=2_048,
)
posterior = reconstructor.reconstruct(
    ReconstructionRequest(
        grid=domain.grid,
        reconstruction_timestamp_utc=current_timestamp,
        measurements=sensing_service.history.records,
    )
)
```

`posterior.mean_ug_m3` y `posterior.standard_deviation_ug_m3` tienen la forma de
la grilla. Las celdas activas contienen predicciones finitas. Las celdas
inactivas se identifican explícitamente mediante `posterior.active_mask` y
contienen `NaN`, no concentraciones inventadas. Los metadatos de observaciones
informan las claves ordenadas de los registros usados en el ajuste y los
conteos de filtrado. El ruido declarado se consulta en los registros
`MeasurementRecord` referenciados, sin copiarlo en un arreglo paralelo de
metadatos. Los diagnósticos indican si se realizó el ajuste, la función de
covarianza configurada, el término efectivo de estabilización numérica y el
logaritmo de la verosimilitud marginal.

## Costo computacional

El ajuste exacto del proceso gaussiano requiere aproximadamente tiempo `O(n^3)`
y memoria `O(n^2)` para `n` observaciones aceptadas. La predicción se realiza
en bloques deterministas de celdas activas para controlar la memoria temporal
sin aproximar el posterior. Por ello, el escenario o la capa de actualización
secuencial debe seleccionar y documentar un conjunto de observaciones
computacionalmente viable. El perfil secuencial provisional de R2.3 aplica el
historial evaluado de seis horas; este módulo de reconstrucción individual no
impone esa política de forma implícita a otros usos.

`prediction_chunk_size` es un parámetro de ejecución cuyo valor predeterminado
es `2048`. Limita el número de celdas activas entregadas en cada llamada de
predicción para controlar la memoria temporal. No cambia el posterior que se
pretende calcular ni es un parámetro científico o calibrado del escenario. Se
puede elegir otro tamaño de bloque positivo según el equipo utilizado.

## Limitaciones

- La desviación estándar posterior representa incertidumbre del modelo, no error
  de predicción medido.
- La función de covarianza temporal representa envejecimiento estadístico, pero
  no incorpora la advección producida por el viento.
- Las escalas de Matérn y la distribución previa requieren calibración o una
  justificación explícita para el escenario sintético.
- La inclusión opcional de registros `CLIPPED` trata sus valores limitados como
  observaciones gaussianas, no mediante una verosimilitud para datos censurados.
- Una media posterior gaussiana puede ser negativa; el estimador no la recorta
  silenciosamente, pues eso alteraría el modelo probabilístico ajustado.
- La calibración histórica del GP excluye por completo la estación objetivo,
  incluidas sus lecturas pasadas y su aporte a la distribución previa de esa
  partición. La reproducción histórica (*replay*) final utiliza todas las
  lecturas fijas observadas elegibles disponibles en cada instante.
- El error del mapa completo y la cobertura de incertidumbre se evalúan frente
  a campos sintéticos retenidos, disponibles solo para el evaluador, nunca
  frente a una superficie interpolada a partir de las estaciones históricas.
