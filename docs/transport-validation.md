# Verificación numérica del transporte — R1.3

## Alcance

Los ensayos comprueban el [modelo de transporte](transport-model.md) con grillas
métricas e inputs sintéticos deterministas. Comparan procesos aislados con
referencias analíticas y verifican conservación, estabilidad y sensibilidad al
refinamiento. No establecen exactitud predictiva para la atmósfera de Lima.

## Reproducción

Ejecutar los ensayos numéricos:

```bash
uv run pytest tests/tesis/r1_3/test_transport_benchmarks.py
```

Ejecutar las comprobaciones complementarias de contratos y fuentes:

```bash
uv run pytest tests/software/simulation/test_transport.py
```

Las entradas de estos casos son locales y no requieren descargar una red OSM.
Los [conjuntos de pruebas](../tests/README.md) se distinguen de los experimentos
de discretización del dominio geográfico completo.

## Ensayos numéricos

Los nombres corresponden a funciones de
[`test_transport_benchmarks.py`](../tests/tesis/r1_3/test_transport_benchmarks.py).

| Proceso | Caso | Criterio |
| --- | --- | --- |
| Advección | `test_advection_moves_gaussian_centroid_with_wind` | Cuatro direcciones cardinales; error longitudinal y transversal no mayor de 1 m. |
| Difusión | `test_diffusion_conserves_mass_and_matches_gaussian_variance_growth` | Masa conservada y aumento de varianza dentro del 8 % de `2 K t`. |
| Salida de frontera | `test_open_boundary_records_outflow` | Salida positiva, entrada nula y residual relativo menor de 10⁻¹². |
| Entrada en dominio irregular | `test_irregular_boundary_admits_only_expected_regional_inflow` | Flujo según cálculo independiente por caras; celdas inactivas en cero y residual menor de 10⁻¹⁰. |
| Deposición seca | `test_dry_deposition_matches_exponential_solution` | Masa final y remoción acordes con decaimiento exponencial, tolerancia relativa 10⁻¹². |
| Deposición húmeda | `test_wet_deposition_depends_on_precipitation` | Ausencia de remoción sin lluvia y decaimiento exponencial con precipitación. |
| Procesos combinados | `test_automatic_substeps_keep_full_budget_balanced_and_nonnegative` | Signos de aportes y remociones, concentraciones no negativas y residual menor de 10⁻¹². |
| Contrafactual | `test_counterfactual_line_event_changes_only_source_inputs` | Forzamiento y fronteras iguales; diferencia atribuida a la fuente del evento. |
| Refinamiento temporal | `test_timestep_refinement_produces_convergent_fields` | Disminución de diferencias y diferencia fina-intermedia menor del 3 % de la norma del campo fino. |

## Contratos complementarios

El conjunto de software comprueba unidades y formas de arreglos, inmutabilidad,
validez del estado y de los proveedores, identidad de fuentes y sus aportes por
intervalo, distribución de emisiones viales, convención del viento y conservación
entre caras. También comprueba estados sin procesos, determinismo, separación
de cambios meteorológicos, fuentes y fronteras, y utilidades de visualización.

Las comprobaciones funcionales de medición, trayectorias y ejecución integrada
están en [`tests/tesis/r1_3/`](../tests/tesis/r1_3/).

## Interpretación de tolerancias

Las tolerancias de desplazamiento y varianza admiten el error de aproximación
del esquema upwind de primer orden y de la difusión centrada. Los balances de
masa comprueban contabilidad numérica, por lo que emplean tolerancias mucho
menores. Un residual próximo a la precisión numérica no implica una exactitud
atmosférica equivalente. El tiempo de ejecución no es criterio de aceptación
porque depende del equipo utilizado.
