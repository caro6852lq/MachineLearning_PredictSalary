# Machine Learning Zoomcamp – Salary Prediction Project

## 1. Problem description

The goal of this project is to build a **machine learning model to predict salaries** (in ARS) based on personal and job-related characteristics of workers in Argentina.

This type of model can be useful for:

* Understanding which factors are most strongly associated with salary levels
* Supporting **benchmarking and compensation analysis**
* Helping HR, analysts, or individuals to estimate expected salary ranges

This project was developed as part of the **Machine Learning Zoomcamp** and follows its recommended structure, tooling, and reproducibility requirements.

---

## 2. Business objective

From a business perspective, predicting salaries helps:

* Identify **market salary ranges** for different profiles
* Reduce information asymmetry in salary negotiations
* Support data-driven decisions in compensation planning

It is important to note that **salary is influenced by many unobserved factors** (company policies, negotiation skills, economic context, inflation, etc.), which limits the maximum achievable accuracy of the model.

---

## 3. Dataset

The dataset comes from the **Sysarmy Salary Survey**, a well-known and widely used survey in the Argentine tech community.

**Sysarmy** is a community of IT professionals in Argentina that periodically publishes anonymous salary surveys, providing valuable insight into:

* Job roles
* Seniority
* Technologies
* Work modality (remote, hybrid, on-site)
* Salaries

All salaries are expressed in **ARS (Argentine pesos)**.

---

## 4. Features

The model uses a subset of available variables, such as:

* Job role / position
* Seniority level
* Years of experience
* Work modality
* Education level
* Technologies

### Potential improvements

The predictive power could be improved by incorporating:

* Company size
* Industry / sector
* Type of contract
* Dollarized vs peso salary
* Adjustments for inflation or time index
* More detailed technology stack variables

---

## 5. Model

The final model is a **regression model** trained to predict salary.

### Performance

* **R² ≈ 0.466**

This means the model explains around **47% of the variance** in salaries.

### Why the model is not a perfect predictor

This performance is expected due to:

* High salary dispersion
* Unobserved variables
* Self-reported survey data
* Market volatility in Argentina

In this context, an R² of ~0.47 is **reasonable for a real-world salary prediction problem**.

---

## 6. Project structure
The project includes these main artifacts (not necessarily in that order):
```
.
├── Data/
│   └── raw_data.csv
├── notebooks/
│   └── exploration.ipynb
├── model/
│   ├── train.py
│   └── model.bin
├── app/
│   └── predict.py
├── Dockerfile
├── requirements.txt
└── README.md
```

---

## 7. Reproducibility

To reproduce this project locally:

### 1. Clone the repository

```bash
git clone https://github.com/caro6852lq/MachineLearning_PredictSalary.git
cd MachineLearning_PredictSalary
```

### 2. Create a virtual environment and install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python model/train.py
```
This will generate the trained model artifact.

### 4. Testing locally with test.py
In addition to calling the API manually, you can test the model locally using the provided test.py script. This script sends a POST request to the /predict endpoint with a sample worker profile.

Steps:

1. Start the API locally:

   ```bash
   python predict.py
   ```
2. In another terminal, run:

   ```bash
   python test.py
   ```
---

## 8. Containerization with Docker

Build the Docker image:

```bash
docker build -t salary-predictor .
```

Run the container:

```bash
docker run -it --rm -p 9696:9696 salary-predictor
```

The service will be available at:

```
http://localhost:9696
```

---

## 9. Deployment

The application is deployed using **Fly.io**.

### Demo video

A short demo video showing the application running on Fly.io is available here:

👉 **(https://github.com/caro6852lq/MachineLearning_PredictSalary/tree/main/Video%20App%20Running)**

---

## 10. API usage example

Example request:

```json
{
    "dedicacion": "Full-Time",
     "tipo_contrato_agrupado": "Staff",
     "anos_de_experiencia": 5,
     "seniority": "Senior",
     "modalidad_de_trabajo": "100% remoto",
     "tamanio_empresa": "De 101 a 200 personas",
     "region": "NOA",
     "genero_agrupado": "Mujer Cis",
    "puesto_agrupado": "BI Analyst / Data Analyst ",
    "recibis_algun_tipo_de_bono": "No",
    "sueldo_dolarizado": "False",
    "exp_por_edad": 0.0,
    "ratio_cargo": 0,
    "movilidad":0
}

```

Example response:

```json
{
  "predicted_salary_ars": 1234567 -- Not is the example exactly
}
```

---

## 11. Limitations

* Model should not be used for **individual salary decisions**
* Best used as a **reference or exploratory tool**
* Sensitive to survey bias and missing variables

---

## 12. Future work

* Feature engineering
* Hyperparameter tuning
* Time-aware modeling (inflation adjustment)
* Model comparison

---

## 13. Author

**Carolina Vergara**
Machine Learning Zoomcamp Student

## Input data format

The API expects a JSON object with the following fields. Categorical values must match those seen during training (examples provided below).

| Field                      | Type    | Description / Examples                         |
| -------------------------- | ------- | ---------------------------------------------- |
| dedicacion                 | string  | "Full-Time", "Part-Time"                       |
| tipo_contrato_agrupado     | string  | "Staff", "Freelance"                           |
| anos_de_experiencia        | integer | Years of experience                            |
| seniority                  | string  | "Junior", "Senior"                             |
| modalidad_de_trabajo       | string  | "100% remoto", "Híbrido (presencial y remoto)" |
| tamanio_empresa            | string  | "De 2001a 5000 personas", "De 101 a 200 personas"|
| region                     | string  | "Pampeana", "NOA"                              |
| genero_agrupado            | string  | "Mujer Cis", "Varón Cis", etc.                 |
| puesto_agrupado            | string  | "Data Scientist", "BI Analyst / Data Analyst" |
| recibis_algun_tipo_de_bono | string  | "Si", "No"                                     |
| sueldo_dolarizado          | string  | "True", "False"                                |
| exp_por_edad               | float   | Engineered feature                             |
| ratio_cargo                | float   | Engineered feature                             |
| movilidad                  | integer | 0 or 1                                         |


