# Movilidad de los sensores móviles

## Función

El contrato de movilidad documenta y aplica las restricciones de las posiciones
de sensores móviles suministradas externamente. Valida posiciones; no
selecciona rutas, acciones, recompensas ni una política de navegación.

`MobilityPolicy` es inmutable durante una ejecución. Para cambiar una condición
experimental se crea otra instancia de la política. Así se conserva la
configuración de una ejecución existente sin impedir que experimentos
posteriores seleccionen otros valores.

## Política de movilidad vial

Construir `MobilityPolicy(road_tolerance_m=...)` con una tolerancia explícita,
en metros, para la distancia entre la posición y la vía. Este valor no tiene
un valor científico predeterminado: el escenario debe justificar por qué la
tolerancia elegida es adecuada. `SpatialDomain` proporciona el grafo OSM
`drive` proyectado y su margen exterior, por lo que la política no repite esa
configuración.

`maximum_speed_m_s` es opcional y no tiene un valor científico predeterminado.
Si se deja en `None`, no se valida la rapidez. Un escenario que seleccione una
rapidez máxima debe registrar su justificación, en lugar de tratar un valor de
prueba como una constante de Lima.

Las distancias se expresan en metros, el tiempo transcurrido en segundos y la
rapidez en metros por segundo, como indican los nombres de los parámetros.
Las trayectorias inválidas se rechazan; las posiciones nunca se ajustan
automáticamente a una vía ni se reemplazan.

## Límite de estudio y margen de la red vial

La grilla de concentración permanece ligada al límite de estudio configurado.
La red vial se descarga sobre un rectángulo WGS84 ligeramente mayor para que
un vehículo pueda seguir calles conectadas justo fuera de la grilla. El valor
inicial configurable de `road_network_buffer_m` es 50 m.

El límite sur predeterminado del estudio se extiende hasta `-12.079` para que
la estación fija de Campo de Marte no quede desproporcionadamente cerca del
borde sur. Las coordenadas fijas conservan las posiciones de sus edificios
reales según el producto de origen; cambiar el límite de estudio nunca modifica
la coordenada de una estación.

El margen de 50 m de la red vial y la tolerancia de posición respecto de una
vía son distintos:

- el margen, definido por el dominio, controla cuánto se extienden la red de
  movilidad descargada y los recorridos permitidos fuera del dominio de
  concentración;
- la tolerancia controla a qué distancia puede estar una posición suministrada
  de la geometría de una vía OSM.

Ambos son parámetros de configuración, no constantes científicas calibradas.

## Trayectorias suministradas

`TimestampedTrajectoryProvider` recibe registros inmutables `TrajectoryPoint`
para un sensor. Los tiempos deben ser estrictamente crecientes y las
coordenadas, finitas en el sistema de referencia de coordenadas (CRS) proyectado
del dominio. Cada instante programado de medición debe tener una posición
registrada explícita. El proveedor no interpola, extrapola, ajusta a una vía
ni corrige una posición ausente.

Cada posición suministrada se comprueba frente a la tolerancia vial elegida y
al `road_network_buffer_m` del dominio. La rapidez máxima se comprueba solo si
el escenario la ha configurado. No hay una regla de separación temporal máxima
entre posiciones: su disponibilidad se verifica directamente en los instantes
programados de medición.

El proveedor tiene un `provider_id`, utilizado como `position_source_id` en
cada `MeasurementRecord` procesado en un instante programado. Su valor
identifica al proveedor, no al contenido de un archivo externo de trayectoria.
Cuando están disponibles, un `resource_id` de entrada opcional y una huella
digital de contenido (*hash*) identifican ese archivo una sola vez en el paquete
de la ejecución o del entorno. El proveedor no mantiene estado mutable después
de la validación. Cada posición procesada se conserva en el historial
cronológico de mediciones, de modo que los intentos válidos y los que quedan
fuera del soporte no requieren un historial mutable duplicado `used_positions`.
