# Modelo de transporte de PM2.5

## Representación y unidades

`simulation/transport.py` implementa un modelo de especie única, verticalmente
promediado, con un método de volúmenes finitos en NumPy. La grilla es horizontal
y bidimensional. La masa de cada celda se relaciona con su concentración por:

```text
masa [µg] = concentración [µg/m³] × área [m²] × altura efectiva [m]
```

Cada escenario declara altura efectiva de mezcla, difusividad horizontal
(`m²/s`), velocidad de deposición seca (`m/s`) y coeficiente de lavado húmedo
(`1/mm`). Los valores de una demostración sintética no son parámetros físicos
calibrados de Lima.

## Entradas independientes

- `AmbientBaseline`: inicialización de concentración y procedencia.
- `AtmosphericForcing`: viento y precipitación, sin concentración de PM2.5.
- `RegionalBoundaryForcing`: concentración externa en la frontera de entrada.
- Proveedores de fuentes: aportes de masa por intervalo, con identidad, rol y
  procedencia explícitos.

La dirección meteorológica indica de dónde proviene el viento, en grados desde
el norte y sentido horario. Las componentes son `u = −v sin(θ)` hacia el este y
`w = −v cos(θ)` hacia el norte. La precipitación se expresa en `mm/h`.

Una colección vacía de fuentes no crea emisiones implícitas. La frontera cerrada
es el modo predeterminado; `regional_inflow` requiere un proveedor regional y
permite entrada a barlovento y salida a sotavento.

## Fuentes

`PM25SourceProvider` entrega un `PM25SourceContribution`: arreglo de masa por
celda en µg y masa integrada declarada. El integrador numérico comprueba forma,
unidades, valores finitos no negativos, celdas inactivas, identidad y
reconciliación de masa.

`GaussianKernelSource` distribuye masa emitida mediante pesos gaussianos
normalizados en la grilla. Es una fuente sintética de verificación, no un modelo
atmosférico de pluma gaussiana. Centro, anchura, tasa y calendario requieren
justificación para representar una emisión urbana concreta.

`RoadLineSource` distribuye emisiones en proporción a la longitud vial dentro
de cada celda. La tasa es `µg/(m s)`; los segmentos sobre una cara compartida
reparten su contribución entre las celdas adyacentes. Las fuentes pueden tener
tiempos de inicio y fin independientes de las cadencias de sensores.

Ambas fuentes aceptan un `EmissionProfile`: multiplicadores constantes o
lineales por tramos, integrados exactamente sobre cada intervalo activo.
`EmissionProfile.daily` expande horarios locales de Lima para el episodio.
`AreaSource` añade emisiones uniformes por superficie activa, con inicio y fin
explícitos. Los [perfiles y episodios](episode-generation.md) documentan las
unidades, el tratamiento de intersecciones y los escenarios de estrés.

## Integración numérica

Cada subpaso aplica:

1. Inyección de masa de las fuentes.
2. Advección de primer orden con valores de la celda a barlovento (*upwind*) y
   flujos de frontera.
3. Difusión mediante diferencias centradas.
4. Deposición seca y húmeda mediante actualizaciones exponenciales.

La tasa de deposición seca es velocidad de deposición dividida entre altura
efectiva. La húmeda es coeficiente de lavado multiplicado por precipitación y
dividido entre 3 600. Las caras internas comparten flujos para conservar masa;
el flujo difusivo externo es nulo. Las celdas inactivas permanecen en cero.

El integrador numérico divide los intervalos según una condición combinada de
estabilidad advectiva y difusiva. El aporte de fuentes se inyecta al inicio del
subpaso.
El [motor integrado](simulation-orchestration.md) separa los intervalos en las
transiciones conocidas de fuentes, meteorología y fronteras.

`MassBudget` registra masa inicial, emitida, entrada, salida, deposiciones,
masa final y residual. Los aportes emitidos también se atribuyen por fuente.

## Demostración y limitaciones

La demostración de `simulation/demo.py` usa geometrías OSM de las avenidas
Túpac Amaru, Javier Prado Este y Circunvalación como fuentes rutinarias, y
Alfonso Ugarte como evento contrafactual de la segunda hora. Las tres horas
cambian viento y precipitación para visualizar la dinámica. Sus emisiones y
parámetros físicos son sintéticos, no estimaciones de tráfico real.

La altura efectiva y difusividad son constantes dentro de cada escenario. El
esquema upwind introduce difusión numérica. No se resuelven química, perfiles
verticales, edificios, vehículos individuales ni otras especies de partículas.
La [verificación numérica](transport-validation.md) documenta referencias y
tolerancias; la [evaluación del estimador](historical-calibration.md) cumple una
función distinta.
