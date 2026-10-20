# Motor integrado de simulación — R1.3

## Propósito

El motor completo combina transporte de PM2.5, meteorología con procedencia
explícita, condiciones de frontera regionales, fuentes de emisión, sensores
fijos y posiciones móviles suministradas para reproducción de trayectorias
(*replay*) en una misma línea temporal UTC con zona horaria. No reconstruye
mapas, selecciona rutas, invoca Gymnasium ni expone el campo oculto mediante
las mediciones públicas.

`iter_simulation` sigue siendo una utilidad de transporte de bajo nivel.
`run_simulation` integra los componentes de R1.3 cuando se requiere evidencia
del campo y de las mediciones.

## Modelo temporal

Cada ejecución tiene un inicio UTC con zona horaria y segundos transcurridos
no negativos. Las entradas consultadas en el tiempo transcurrido `t` rigen el
intervalo semiabierto `[t, next_t)`. El motor divide el transporte en:

- transiciones meteorológicas;
- transiciones de las condiciones de frontera regionales;
- inicios, finales y puntos de cambio de los perfiles temporales de fuentes;
- instantes programados de medición de sensores fijos o móviles;
- instantes de salida de concentración;
- final del episodio.

Los subpasos de estabilidad numérica son internos a `PM25Transport.step`. Los
puntos de trayectoria que no coinciden con instantes de medición no dividen el
transporte.

`SimulationRunConfig` acepta un intervalo regular
`concentration_output_interval_s`, instantes explícitos y ordenados
`concentration_output_times_s`, o ambos. Los instantes duplicados entre las
dos fuentes se resuelven como un único punto de salida, y el final del episodio
siempre se conserva como salida. Los tiempos explícitos permiten obtener un
campo completo en un instante irregular solicitado, sin exponer ni guardar
cada subpaso interno de estabilidad.

Al llegar al límite de un intervalo, el motor registra el campo si está
configurado para ese instante, muestrea los sensores programados en orden de
identificador y forma un lote con los registros numéricos. Los sensores
programados en `t=0` muestrean el campo inicial antes de las emisiones o el
transporte del intervalo. Los intentos `DROPPED` y `OUTSIDE_ACTIVE_GRID`
permanecen en el historial, pero no entran en los lotes aceptados; los registros
numéricos `ACCEPTED` y `CLIPPED` sí entran.

## Validación previa de la reproducción de trayectorias

El motor de R1.3 valida la ejecución completa antes de avanzar el transporte:

- la ventana meteorológica tiene el mismo inicio UTC y duración;
- cada proveedor expone sus transiciones o es explícitamente constante;
- el modo de frontera y el calendario de condiciones regionales son coherentes;
- las referencias a sensores y estaciones son válidas;
- cada instante móvil programado tiene una posición explícita;
- los calendarios de fuentes y salidas contienen tiempos finitos y ordenados.

Este contrato reproduce trayectorias suministradas. Un agente interactivo de
navegación no puede declarar posiciones futuras todavía desconocidas y queda
fuera del alcance de R1.3. En el futuro, un agente podría reutilizar la misma
mecánica de avance entre eventos sin cambiar el significado de las mediciones
de R1.3.

## Meteorología y linaje de los datos

Las ventanas históricas y las explícitamente sintéticas implementan el mismo
contrato operacional. Las históricas conservan la identidad y la huella digital
de contenido (*hash*) del recurso, la estación, la cobertura UTC, la política
de precipitación y el linaje estructurado por variable. Las sintéticas
conservan su proveedor y configuración sintéticos. Las ventanas con
perturbaciones sintéticas del viento conservan además la base histórica y su
política y linaje de precipitación sin cambios; el viento generado nunca se
identifica como observado.

La estructura compacta `AtmosphericForcing` enviada al transporte contiene
viento y precipitación. El linaje estructurado de diagnóstico se mantiene
separado para que el motor no tenga que interpretar cadenas de procedencia
ni repetir el control de calidad del productor.

## Paquete de la ejecución

`write_run_bundle` escribe en un destino nuevo y rechaza sobrescribir uno
existente. Un paquete completo contiene:

```text
<run-id>/
|-- manifest.json
|-- concentration.npz
|-- measurement_attempts.jsonl
|-- measurement_batches.jsonl
|-- diagnostics.jsonl
`-- checksums.json
```

`concentration.npz` almacena los tiempos transcurridos, las marcas de tiempo
UTC, el campo con dimensiones `[time, row, column]` —tiempo, fila y columna—,
los centros de la grilla, la máscara activa y el sistema de referencia de
coordenadas (CRS). Los intentos conservan las posiciones móviles exactas
utilizadas. Los lotes referencian intentos aceptados en lugar de crear una
segunda fuente de mediciones. El manifiesto registra el descriptor de entrada
de la trayectoria; no se crea un historial mutable separado de posiciones
utilizadas.

Los episodios generados registran además `episode_generation` en el mismo
manifiesto: partición, configuración, semilla y ventana histórica seleccionada.
La [generación de episodios](episode-generation.md) describe la interfaz de
línea de comandos y los supuestos del modelo.

`checksums.json` se escribe al final e incluye las huellas SHA-256 de todos
los demás archivos. `verify_run_bundle` identifica archivos ausentes o
modificados. Así se obtiene evidencia finalizada cuyas alteraciones pueden
detectarse; no se afirma que sea inmutable a nivel del sistema operativo ni
idéntica byte a byte entre distintas plataformas de software.

## Ejemplo completo

Ejecutar desde la raíz del repositorio:

```bash
uv run lima-pm25-core
```

El ejemplo utiliza un escenario explícitamente sintético de demostración del
funcionamiento del motor, cuatro sitios fijos de PM2.5, una trayectoria móvil
suministrada y restringida a la red vial con mediciones cada cinco minutos, y
ruido móvil opcional controlado por semilla. Escribe un paquete en:

```text
resultados/r1_3_simulacion/<run-id>/
```

El ejemplo de consola carga `--study-config`, cuyo valor predeterminado es
`config/study_environment.json`, y utiliza el calendario de salidas seleccionado
en esa configuración. El experimento de sensibilidad conserva su resumen por
separado, no un segundo conjunto de paquetes completos. Las pruebas de software
se ejecutan con pytest; no constituyen otro producto de simulación.

Repetir la misma configuración reutiliza el paquete ya verificado, en lugar
de sobrescribirlo. El ejemplo demuestra el funcionamiento y el contrato de
ejecución de R1.3; su meteorología, emisiones y coeficientes físicos siguen sin
estar calibrados y no deben citarse como condiciones observadas de Lima.

## Comprobación funcional

El ejemplo integrado utiliza la cadencia de salida seleccionada en la
configuración: 37 campos de 0 a 10 800 s cada 300 s. Los casos de transporte,
sensores, trayectorias y orquestación comprueban las capacidades de R1.3; el total
de pruebas del repositorio incluye también otros resultados. Los casos funcionales
están en [`tests/tesis/r1_3/`](../tests/tesis/r1_3/); los contratos internos se
comprueban en [`tests/software/simulation/`](../tests/software/simulation/).
Véanse el [modelo](transport-model.md) y sus [ensayos numéricos](transport-validation.md).
