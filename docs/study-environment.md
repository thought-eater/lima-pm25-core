# Entorno de estudio y discretización — R1.2

## Dominio y estaciones

La configuración del entorno se encuentra en
[`config/study_environment.json`](../config/study_environment.json). Define el
rectángulo WGS84 con longitud entre −77,05108161742221 y −76,90976879099601 y
latitud entre −12,079 y −11,974250416076465. OSMnx obtiene la red vial `drive`
y la proyecta a un sistema métrico local.

La grilla seleccionada tiene celdas de 200 m, 59 filas, 78 columnas y 4 459
celdas activas. El [dominio conservado](../resultados/r1_2_entorno/domain/)
incluye el grafo GraphML, la grilla NPZ, los metadatos y el mapa georreferenciado.

Las coordenadas de Campo de Marte, Ceres, Santa Anita y San Juan de Lurigancho
proceden del producto PM2.5. Se conserva la posición proyectada exacta de cada
estación y se registra por separado su celda contenedora. Las coordenadas del
producto meteorológico no sustituyen a las de PM2.5. Véase el
[catálogo y contrato de medición](sensing.md).

La red vial añade un margen exterior configurable de 50 m para trayectos por
calles conectadas. Ese margen no amplía la grilla de concentración ni equivale
a la tolerancia de posición respecto de una vía. La tolerancia y la velocidad
máxima opcional son parámetros explícitos de la [política de movilidad](mobile-mobility.md).

## Interpretación espacial y vertical

Los arreglos usan índices `[fila, columna]`: la celda `(0, 0)` está al suroeste,
las columnas crecen hacia el este y las filas hacia el norte. Las figuras
ráster usan `origin="lower"`. `GridDefinition.xy_to_cell()` y `cell_center()`
realizan la conversión entre posiciones y celdas.

La grilla resuelve dos dimensiones horizontales. Cada celda representa una
columna de aire verticalmente promediada:

```text
volumen efectivo = área horizontal × altura efectiva de mezcla
```

La concentración se expresa en µg/m³. La altura constante de 200 m es una
aproximación provisional aceptada para el alcance de la tesis, según lo
comunicado por el autor; no es una altura atmosférica medida o calibrada para
Lima. El modelo no resuelve capas verticales, edificios ni cañones urbanos.

## Escenarios y campo de referencia

- **Sintéticos:** el simulador conoce el campo completo. Son la fuente principal
  prevista de episodios de entrenamiento y permiten evaluación controlada.
  El estimador recibe únicamente mediciones; el campo se reserva al evaluador.
- **Reproducción histórica (*replay*):** utiliza meteorología y observaciones
  fijas elegibles de sus productos, junto con mediciones móviles explícitamente
  simuladas. Ceres conserva su sitio, pero no aporta observaciones de PM2.5.
  El replay es una aproximación modelada a condiciones históricas, no un
  despliegue físico.
- **Estrés y contrafactuales:** identifican condiciones fuera de los rangos
  plausibles o cambios deliberados. Un miembro de control es una comparación,
  no verdad histórica; cada contrafactual modifica solo sus factores declarados.

Las observaciones dispersas no se interpolan a una superficie completa para
presentarla como referencia independiente. Los episodios nominales requieren
restricciones basadas en evidencia de Lima; véase la
[calibración y evaluación](historical-calibration.md).

## Interpretación temporal

El motor avanza por eventos y combina distintas cadencias: meteorología,
transiciones de fuentes y fronteras, mediciones, salidas solicitadas y subpasos
de estabilidad. Las emisiones aportan masa por intervalo y pueden cambiar en
segundos arbitrarios, independientemente de las mediciones.

Se pueden solicitar campos a intervalos regulares, en una lista de instantes
explícitos o mediante ambos mecanismos. Obtener un campo en un instante supone
integrar numéricamente hasta él, no disponer de una solución analítica continua.

El experimento seleccionó 300 s para conservar campos completos. La cadencia
móvil de cinco minutos es un parámetro provisional separado; no es una
resolución temporal universal.

## Experimentos de discretización

El [protocolo](../config/discretization_protocol.json) declara candidatos,
cantidades y tolerancias. El [informe numérico](../resultados/r1_2_entorno/discretization_evidence.json)
conserva las métricas de todas las alternativas y la selección resultante.

### Comparación horizontal

Se comparan 50, 100 y 200 m bajo condiciones comunes de dominio, red vial,
escenario, semilla, fuentes y tiempos. La regla selecciona la alternativa más
gruesa aceptable frente a la siguiente más fina, exigiendo ambas comparaciones.

| Cantidad | Tolerancia | 100 frente a 50 m | 200 frente a 100 m |
| --- | ---: | ---: | ---: |
| Concentraciones en estaciones | 10 % | 0,37 % | 2,53 % |
| Media espacial | 5 % | 0,067 % | 0,050 % |
| Masa significativa del evento | 5 % | 0,0052 % | 0,0107 % |
| Desplazamiento del centroide | 100 m | 12,77 m | 35,14 m |

Ambas comparaciones conservaron el orden control/evento y residuales relativos
de masa del orden de 10⁻¹⁵. La masa adicional y el centroide se comparan cuando
el efecto supera el 1 % de su máximo; las contribuciones residuales no sustentan
comparaciones porcentuales estables. Richardson/GCI se reporta solo cuando la
serie presenta el comportamiento de convergencia requerido.

La selección de 200 m demuestra suficiencia para el escenario y las cantidades
ensayadas, no exactitud atmosférica ni optimalidad para cualquier fenómeno.

### Comparación temporal

Se comparan salidas de 300, 900 y 3 600 s. El protocolo admite hasta 3 % de
diferencia relativa L1 en instantes compartidos y 5 % de pérdida del máximo y
de diferencia de la trayectoria entre salidas. Las alternativas de 900 y 3 600 s
conservaron los instantes comunes y máximos, pero excedieron el criterio de
trayectoria significativa del evento. Se seleccionaron 300 s.

## Reproducción

Desde la raíz del repositorio, generar el mapa:

```bash
uv run lima-pm25-core --domain-only
```

Repetir el experimento en un destino nuevo:

```bash
uv run lima-pm25-core --study-sensitivity \
    --sensitivity-output cache/reproduccion/discretization_evidence.json
```

El exportador rechaza sobrescribir un informe existente. Conserva métricas por
alternativa; no requiere publicar campos completos de todos los candidatos.
El ejemplo integrado de R1.3 conserva una ejecución completa por separado.

La opción `--study-evidence` genera, cuando se necesita, un paquete integral de
configuración, dominio, familias y hashes bajo `resultados/r1_2_entorno/paquetes/`.
El [índice de productos](../resultados/README.md) enlaza la evidencia seleccionada
sin duplicar esos archivos.
