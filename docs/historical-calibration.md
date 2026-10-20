# Calibración histórica y evaluación retenida

## Propósito

El procedimiento de R2.1–R2.2 separa tres actividades:

1. calibrar la plausibilidad de escenarios nominales para Lima;
2. seleccionar hiperparámetros del proceso gaussiano con observaciones históricas;
3. evaluar mapas completos de concentración e incertidumbre sobre campos
   sintéticos retenidos.

La reproducción histórica (*replay*) usa después la configuración congelada y
todas las observaciones fijas elegibles disponibles. No es un piloto físico ni
dispone de un campo histórico completo observado.

## Insumos suministrados

```text
data/pm25_lima_2025_2026_imputed.csv
data/var_meteorologicas_lima_2025_2026_imputed.csv
```

`load_historical_pm25_csv` preserva cada fila de PM2.5, sus coordenadas, marca
de tiempo, valor ausente o presente y `PM2_5_SOURCE`. Solo un valor presente con
`PM2_5_SOURCE=senamhi_observed` puede convertirse en medición pública, valor
objetivo de calibración o referencia para calcular métricas. Los valores
imputados permanecen como evidencia del producto, pero no se presentan al
proceso gaussiano (GP, por sus siglas en inglés) como observaciones reales.

La meteorología conserva el contrato descrito en
[`historical-weather.md`](historical-weather.md): viento y precipitación
observados o imputados por el productor pueden forzar el transporte, y `PP_IFS`
requiere una política explícita.

### Cobertura comprobada

El perfil de solo lectura del producto PM2.5 suministrado encontró:

- 117 936 filas entre 2025-01-01 05:00 UTC y 2026-07-01 04:00 UTC;
- 85 016 valores `senamhi_observed`;
- nueve identificadores de estación en el producto completo;
- tres series observadas dentro del catálogo de estudio: Campo de Marte, Santa
  Anita y San Juan de Lurigancho;
- Ceres conserva su sitio, pero sus 13 104 horas son `missing_structural` y no se
  fabrica una cuarta lectura;
- 11 045 horas con las tres series observadas del catálogo y meteorología
  cubierta de Campo de Marte.

## Partición histórica

`build_historical_partition_manifest` aplica una regla cronológica reproducible.
El perfil provisional usa 70 % inicial para calibración, 30 % final para replay,
ocho ventanas de calibración y cuatro de replay de 24 horas, hasta 24 horas
previas de contexto y al menos tres estaciones observadas por hora.

Las ventanas se distribuyen sin solapamiento y se seleccionan por cobertura, no
por desempeño. El manifiesto registra huellas digitales de contenido (*hashes*)
de los CSV, límites UTC, conteos por estación, política de observación y método
de selección. Puede guardarse una sola vez mediante
`write_historical_partition_manifest`.

Estos valores son decisiones operacionales provisionales del estudio. Cualquier
modificación debe congelarse antes de abrir evaluaciones retenidas o resultados
finales.

## Calibración del proceso gaussiano

La calibración utiliza exclusión completa por estación (`leave_one_station_out`).
En cada turno, ninguna concentración de la estación objetivo entra al GP, incluidas
sus lecturas anteriores. La mediana y desviación estándar previas se calculan una
sola vez con las otras estaciones de las ventanas de calibración y se reutilizan
entre candidatos. Las entradas de cada predicción se restringen al historial
permitido y a marcas de tiempo anteriores o simultáneas. Ceres no genera objetivos.
Esta retención no se usa durante el replay final.

El perfil candidato inicial combina historiales de 6, 12 o 24 horas, escalas
espaciales isotrópicas de 500, 1 000, 2 000 o 4 000 m y escalas temporales de 1,
3, 6 o 12 horas. Después de la selección, el modelo operacional utiliza la mediana
y desviación estándar poblacional de todas las estaciones de calibración; estos
parámetros globales no se utilizan para predecir las estaciones excluidas.
El RMSE agregado de las
estaciones retenidas es el criterio principal; MAE, error de cobertura y ancho
medio son desempates.

`gp_calibration.json` conserva el método, los parámetros previos por estación
excluida y las métricas agregadas y por estación de cada candidato. El resultado
es una validación para selección de parámetros; no es una prueba histórica
independiente de esa selección.

La predicción de calibración se realiza directamente en la coordenada de la
estación mediante el mismo GP usado para la grilla, evitando calcular miles de
celdas para un único valor objetivo. El producto no declara incertidumbre
instrumental por lectura; la desviación de ruido histórica es una entrada
explícita pendiente de sustento y no se inventa dentro del adaptador.

## Plausibilidad de escenarios

`build_lima_plausibility_profile` resume únicamente PM2.5 observado y la
meteorología admitida. El perfil real inicial obtuvo:

| Variable | Mediana | P01 | P99 | Máximo |
| --- | ---: | ---: | ---: | ---: |
| PM2.5 (`ug/m3`) | 21,4 | 4,7 | 90,885 | 344,5 |
| Viento (`m/s`) | 0,9 | 0,0 | 2,5 | 3,3 |
| Temperatura (`°C`) | 20,3 | 14,5 | 28,9 | 32,0 |
| Humedad relativa (`%`) | 86 | 60 | 98 | 100 |
| Precipitación (`mm/h`) | 0,0 | 0,0 | 0,0 | 1,3 |

PM2.5, viento, temperatura y humedad usan provisionalmente P01–P99 como
envolvente nominal. La precipitación es cero-inflada: como P99 también es cero,
usa mínimo–máximo observado para no clasificar toda lluvia válida como estrés.
Una precipitación torrencial seguiría fuera de la envolvente de Lima.

Los parámetros ausentes de los CSV deben clasificarse como históricamente
restringidos, restringidos por literatura o sintéticos. Obtener concentración
plausible no convierte una tasa de emisión o altura de mezcla en una estimación
de su valor real para Lima.

## Evaluación R2.1

La calibración sintética utiliza el miembro de control; la evaluación usa el
miembro con evento en una ventana retenida. La escala de normalización procede
del PM2.5 histórico de calibración. El GP recibe solo mediciones públicas fijas y
móviles; el evaluador compara después el posterior con el campo oculto.

```text
S = P95 - P05 del PM2.5 observado en la partición histórica de calibración
NMAE = MAE / S
NRMSE = RMSE / S
skill_RMSE = 1 - RMSE_GP / RMSE_prior-only
```

Antes del retenido, el contraste sintético P99-P50 debe alcanzar al menos 20 % de
`S`. Los límites son 100 % de mapas válidos, NMAE <= 0,15, NRMSE <= 0,20 y RMSE
menor que el del modelo basado solo en la media previa. R2 se reporta como
diagnóstico. Una escala no positiva, soporte insuficiente, escenario de
referencia sin contraste, mapa faltante o valor activo no finito produce estado
`invalid`, no `pass`.

## Evaluación R2.2

Cada media posterior se conserva con la desviación estándar del mismo posterior,
grilla, máscara, instante y corte observacional. El perfil provisional exige
incertidumbre finita y no negativa, cobertura histórica rotativa entre 90 % y
98 % para intervalos nominales del 95 %, ancho histórico posterior medio menor
que el promedio de los anchos previos de las mismas predicciones excluidas,
varianza local e integrada no creciente al añadir información móvil en el mismo
instante y crecimiento hacia la incertidumbre previa cuando una observación
envejece sin reemplazo.

La desviación estándar posterior es incertidumbre del modelo, no incertidumbre
atmosférica total. Aun con esa limitación, es la señal observable que un agente
puede usar para seleccionar regiones informativamente valiosas.

## Replay histórico final

El replay combina meteorología y fechas históricas, proveedores del campo
calibrados o restringidos, todas las lecturas fijas observadas elegibles del
catálogo y lecturas móviles explícitamente simuladas.

`compose_historical_replay_batches` conserva por separado la procedencia fija
observada y móvil simulada. Ceres permanece sin lectura histórica. Los intentos
móviles sin valor no entran al GP. `reconstruct_measurement_series` selecciona
solo observaciones anteriores o simultáneas dentro del historial configurado.

El replay no se usa como verdad completa para calcular error espacial. Las
métricas completas pertenecen a las ejecuciones sintéticas retenidas.

## Evidencia

`write_retained_evaluation_bundle` guarda la configuración, las métricas
R2.1/R2.2, un registro estructurado, las matrices NPZ del campo sintético
retenido y de las estimaciones con sensores fijos y con fijos más móviles, solo
las figuras representativas declaradas y las huellas de integridad verificables.
La evidencia numérica completa permanece separada de la cantidad limitada de
figuras necesarias para inspección y presentación.

Para repetir la calibración, copiar `config/estimation_evaluation.json` a
`cache/reproduccion/estimation_evaluation.json` y cambiar en esa copia
`calibration_artifact` a `cache/reproduccion/calibracion/final_calibration.json`.
El destino debe ser nuevo. Ejecutar con una desviación de ruido explícita:

```bash
uv run lima-pm25-core \
    --historical-calibration \
    --historical-noise-stddev 0 \
    --study-config config/study_environment.json \
    --estimation-profile cache/reproduccion/estimation_evaluation.json \
    --historical-output cache/reproduccion/calibracion
```

El valor cero reproduce el perfil técnico congelado, pero no constituye una
selección científica de incertidumbre instrumental. La salida no sobrescribible
se guarda bajo el directorio elegido, identificada por su huella digital. El
artefacto final se escribe en la ruta declarada por el perfil de trabajo.

El perfil actual se encuentra en `config/estimation_evaluation.json` y declara
directamente la ruta del artefacto. La interfaz de línea de comandos conserva
primero el informe histórico, incluida una identificación del procedimiento de
exclusión, y después compara las escalas espaciales sintéticas usando el
historial y la escala temporal elegidos.
La selección obtenida actualiza el perfil existente antes de abrir la evaluación
retenida. El estado histórico y los motivos de incumplimiento se conservan aunque
se generen productos diagnósticos posteriores.
Ver el [índice de productos](../resultados/README.md).

## Resultado con exclusión completa por estación

La ejecución `f68cc53e…` evaluó 48 candidatos sobre 576 objetivos (192 por
estación). La mejor configuración histórica utilizó seis horas de historial,
4 000 m y una hora de escala temporal: MAE 8,8269 µg/m³, RMSE 12,2947 µg/m³ y
cobertura 94,9653 %. La comparación sintética seleccionó 2 000 m, conservando el
historial y la escala temporal elegidos. El perfil final es `d4b9abf1…`.

| Escala espacial sintética (m) | RMSE de calibración (µg/m³) |
| --- | ---: |
| 500 | 2,236518 |
| 1 000 | 2,173184 |
| 2 000 | 2,137188 |
| 4 000 | 2,247166 |

Para esa configuración final de 2 000 m, la evaluación histórica por exclusión
obtuvo MAE 9,8584 µg/m³, RMSE 13,8448 µg/m³ y cobertura 93,4028 %. El ancho
posterior medio fue 51,88208 µg/m³ frente a 51,88519 µg/m³ de los previos de las
particiones. La reducción del 0,006 % cumple estrictamente el criterio de ancho,
pero muestra que las otras estaciones apenas informan esas ubicaciones bajo la
escala espacial seleccionada. Esta cobertura no demuestra reconstrucción precisa.

La evaluación retenida `19010dd4…` conservó 37 mapas completos y obtuvo:

| Indicador | Solo fijas | Fijas y móvil |
| --- | ---: | ---: |
| MAE (µg/m³) | 1,430944 | 1,513393 |
| RMSE (µg/m³) | 3,212838 | 3,131960 |
| NMAE | 0,034522 | 0,036511 |
| NRMSE | 0,077511 | 0,075560 |
| R² | 0,175876 | 0,216845 |
| Mejora del RMSE frente al previo | 12,7292 % | 14,9261 % |

R2.1 y el criterio histórico R2.2 resultaron `pass`. La cobertura sintética fue
99,7297 %, con ancho posterior de 47,53441 µg/m³ y previo de 52,38521 µg/m³;
el diagnóstico sintético de cobertura sigue siendo `fail` por exceso de cobertura.
Los cuatro mapas de replay conservaron nueve lecturas fijas reales y 35 móviles
simuladas, con concentración e incertidumbre válidas en todas las celdas activas.

La curva de envejecimiento abarca cinco horas. La desviación estándar llega
a 11,6990 µg/m³ a una hora, 13,2327 a dos y 13,3559 a tres, acercándose al previo
de 13,3638.

## Repetir la evaluación con parámetros congelados

`--estimation-evaluation` carga el artefacto congelado y no repite la calibración.
`--estimation-profile` y `--estimation-output` seleccionan el perfil y el
directorio de salida. El destino por defecto es
`resultados/r2_1_r2_2_estimacion/evaluacion/<fingerprint>/`; el paquete conserva
juntos `synthetic_retained/`, `r2_evaluation/` e `historical_replay/`.
Cada nueva ejecución utiliza la configuración actual y su propia identidad.

```bash
uv run lima-pm25-core --estimation-evaluation \
    --estimation-profile config/estimation_evaluation.json \
    --estimation-output cache/reproduccion/evaluacion
```

Usar un destino nuevo para cada reproducción. La configuración publicada ya
incluye su calibración final; este comando no vuelve a seleccionar parámetros.

## Figuras y tablas para el capítulo del objetivo 2

El exportador de evaluación deriva las figuras de las matrices que también
utiliza el reporte numérico. Los archivos `frame-*.png` muestran campo sintético
de referencia, concentración estimada y error absoluto, con escalas comunes en
la serie, coordenadas UTM en kilómetros y posiciones de las mediciones realmente
utilizadas. El historial móvil y la medición del instante se distinguen en la
leyenda. Las celdas inactivas se enmascaran al calcular escalas y diferencias.

`mobile-uncertainty.png` compara desviación estándar con estaciones fijas y con
historial fijo+móvil en el instante representativo intermedio. La reducción de
varianza conserva su signo: una disminución es positiva y un aumento sería
negativo. Esta comparación corresponde al historial móvil completo disponible,
no al aporte aislado de la última medición.

`temporal-aging.png` representa un caso controlado con una sola medición ideal,
predicción en su posición exacta y parámetros operacionales constantes. No se
aplica el límite de historial: se aísla la pérdida de correlación temporal, sin
confundirla con la eliminación de una observación después de seis horas. Los
valores y condiciones se conservan en `metrics.json`, sección `temporal_aging`.

El mismo archivo incorpora `map_validation` con cobertura, instantes y contraste
P99−P50 calculado sobre el campo de la simulación utilizada para evaluar, y
`illustrated_mobile_uncertainty` con los valores de la figura seleccionada. Las
métricas R2.1/R2.2 se conservan junto con esos diagnósticos.

El replay exporta `historical_replay/figures/replay-maps.png`, con concentración
e incertidumbre en un instante intermedio y el final. No incluye un supuesto
campo histórico completo de referencia. Ceres aparece como sitio sin observación
utilizada. `metadata.json` incorpora cobertura y conteos únicos de entradas
utilizadas, separando contexto anterior de la ventana (incluido su extremo final
para las mediciones móviles). La ejecución existente utiliza nueve lecturas
fijas observadas y 35 móviles simuladas, sin contexto previo incorporado.

Los exportadores `export_evaluation_figures` y `export_replay_figures` también
permiten actualizar estas presentaciones a partir de los NPZ y las mediciones
originales, sin recalibrar ni reconstruir de nuevo las series. Se debe usar la
grilla y configuración de la ejecución correspondiente. El segundo requiere los
registros históricos elegibles originales además de los intentos móviles; los
intentos fijos simulados no sustituyen esas observaciones.

Las comprobaciones del cálculo de métricas y de los exportadores están en
[`tests/software/estimation/`](../tests/software/estimation/). Las comprobaciones
controladas de respuesta espacial y temporal y de aislamiento del campo están en
[`tests/tesis/r2_1_r2_2/`](../tests/tesis/r2_1_r2_2/). Las cifras de desempeño
proceden de las ejecuciones experimentales enlazadas en el índice de productos.
