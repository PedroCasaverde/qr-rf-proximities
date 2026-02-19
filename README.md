# Quantile Regression using Random Forest Proximities

Replicacion y extension del paper **"Quantile Regression using Random Forest Proximities"** (Li et al., 2024 — BlackRock) sobre 3 datasets distintos.

> **Paper:** Li, M., Sarmah, B., Desai, D., Rosaler, J., Bhagat, S., Sommer, P., & Mehta, D. (2024).
> *Quantile Regression using Random Forest Proximities*. arXiv:2408.02355v1 [stat.ML].

## Integrantes

| Nombre | Dataset | Target |
|--------|---------|--------|
| Pedro Casaverde | Online Retail II | Quantity (unidades vendidas) |
| Gian Carlo Quiroz| Energy Efficiency | Heating Load (Y1) |
| Sherly Salazar| Online News Popularity | log(1 + shares) |

## Que hace este proyecto?

En lugar de predecir un solo numero, este paper propone predecir **el rango completo de posibilidades** usando proximidades de Random Forest. Comparamos 4 metodos:

| Metodo | Descripcion |
|--------|-------------|
| **QRF** | Quantile Regression Forest (Meinshausen, 2006) |
| **RF-GAP** | Geometry- and Accuracy-Preserving proximity (Eq. 4) |
| **OOB** | Out-of-Bag proximity (Eq. 6) |
| **ORIGINAL** | Original proximity (Eq. 5) |

## Estructura del repositorio

```
qr-rf-proximities/
├── README.md
├── run_all_datasets.py          # Script principal — ejecuta los 3 datasets
├── data/                        # Datos
│   ├── online_retail_II.xlsx
│   └── OnlineNewsPopularity.csv
└── plots/                       # Resultados visuales
    ├── online_retail/           # 9 graficos + tablas CSV
    ├── energy_efficiency/       # 9 graficos + tablas CSV
    └── online_news/             # 9 graficos + tablas CSV
```

## Resultados principales

| Criterio | Energy Efficiency | Online Retail | Online News |
|----------|-------------------|---------------|-------------|
| Quantile Loss | RF-GAP gana | RF-GAP gana | Empate |
| MSE | QRF gana | RF-GAP gana | RF-GAP (marginal) |
| Intervalos mas estrechos | RF-GAP | RF-GAP ~ QRF | Todos similares |
| Nivel de ruido | Bajo | Medio | Alto |
| **Ventaja RF-GAP** | **Clara** | **Moderada** | **Minima** |

**Hallazgo clave:** La ventaja de RF-GAP es inversamente proporcional al ruido en los datos.

## Graficos generados (por dataset)

Cada dataset genera 9 visualizaciones:

1. `01_quantile_loss_comparison.png` — Quantile Loss por metodo y cuantil (Table 2)
2. `02_mse_mape_comparison.png` — MSE y MAPE de la mediana condicional (Eq. 9, 10)
3. `03_prediction_intervals_rfgap.png` — Intervalos de prediccion RF-GAP (Fig. 1)
4. `04_width_prediction_intervals.png` — Ancho de intervalos comparado (Fig. 3)
5. `05_prediction_intervals_all_methods.png` — Intervalos de los 4 metodos (Fig. 2)
6. `06_feature_importance.png` — Importancia de features (MDI)
7. `07_coverage_calibration.png` — Calibracion de cobertura
8. `08_conditional_distribution.png` — CDF condicional completa (Eq. 7)
9. `09_split_criteria_comparison.png` — Impacto del criterio de split (Table 3)

## Como ejecutar

### 1. Instalar dependencias

```bash
pip install numpy pandas matplotlib scikit-learn openpyxl
```

### 2. Preparar los datos

Los datasets de **Online Retail II** y **Online News Popularity** ya estan incluidos en la carpeta `data/`.

El dataset de **Energy Efficiency** se descarga automaticamente via `sklearn.datasets.fetch_openml` al ejecutar el script.

### 3. Ejecutar

```bash
python run_all_datasets.py
```

Los graficos se generan automaticamente en `plots/`.

## Metodologia

1. **Grid Search + 5-Fold CV** para optimizar hiperparametros del Random Forest
2. **Evaluacion 5-Fold CV** de los 4 metodos en 7 cuantiles (0.5%, 2.5%, 5%, 50%, 95%, 97.5%, 99.5%)
3. **Analisis de criterio de split** (squared_error vs absolute_error)
4. **Generacion de 9 visualizaciones** por dataset

## Referencia

```bibtex
@article{li2024quantile,
  title={Quantile Regression using Random Forest Proximities},
  author={Li, Mingshu and Sarmah, Bhaskarjit and Desai, Dhruv and Rosaler, Joshua and Bhagat, Snigdha and Sommer, Philip and Mehta, Dhagash},
  journal={arXiv preprint arXiv:2408.02355},
  year={2024}
}
```

## Curso

Machine Learning — Ciclo III — UTEC (2026-I)
