# Datos de entrada

Estos CSV son productos horarios del preprocesamiento de datos SENAMHI
realizado en el proyecto `senamhi-pre-procesamiento` (resultado R1.1).
Se conservan aquí como entradas reproducibles de R1.2–R2.3.

| Archivo | Contenido | Uso |
| --- | --- | --- |
| [pm25_lima_2025_2026_imputed.csv](pm25_lima_2025_2026_imputed.csv) | PM2.5, estaciones, coordenadas y `PM2_5_SOURCE`. | Catálogo, calibración del GP y replay histórico. |
| [var_meteorologicas_lima_2025_2026_imputed.csv](var_meteorologicas_lima_2025_2026_imputed.csv) | Meteorología y procedencia `<VAR>_SOURCE` por variable. | Forzamiento del transporte y restricciones de plausibilidad. |

Los productos preservan observaciones, imputaciones y ausencias distinguidas por
sus columnas de procedencia. Una imputación no se convierte en observación real.
Para calibrar el GP y suministrar mediciones fijas al replay solo se admiten
valores PM2.5 presentes con `PM2_5_SOURCE=senamhi_observed`.

El catálogo del dominio incluye Campo de Marte, Ceres, Santa Anita y San Juan de
Lurigancho. Ceres no tiene valores observados de PM2.5 en el producto; no se
genera una cuarta serie histórica. Las coordenadas de PM2.5 y de meteorología
permanecen vinculadas a su propio producto.

Las fechas y horas originales se interpretan en `America/Lima` y se convierten a
UTC dentro del simulador. El adaptador meteorológico acepta valores observados o
imputados admisibles sin repetir el control de calidad del productor. `PP_IFS`
es precipitación modelada externa y solo se utiliza como respaldo explícito.

Los manifiestos de calibración y evaluación registran los hashes de los CSV.
Véanse el [contrato meteorológico](../docs/historical-weather.md) y el
[procedimiento de calibración](../docs/historical-calibration.md).
