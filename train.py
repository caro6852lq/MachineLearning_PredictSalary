
import pickle
import xgboost as xgb

import pandas as pd ## to import the dataset and analyze it
import matplotlib.pyplot as plt ## for statistical graphs
import seaborn as sns ## for statistical graphs
import numpy as np ## for work with matrices
from sklearn.model_selection import train_test_split ## for split the dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer


# Reading the dataset

url = "https://raw.githubusercontent.com/caro6852lq/MachineLearning_PredictSalary/main/Data/2025.2%20-%20Sysarmy%20-%20Encuesta%20de%20remuneraci%C3%B3n%20salarial%20Argentina%20-%20Sysarmy%20-%20sueldos%20-%202025.02CLEAN.csv"

df = pd.read_csv(url, skiprows=9) ## The file has header and rows that do not contain data

# Selected columns
columnas = [
    'donde_estas_trabajando', 
    'dedicacion', 
    'tipo_de_contrato',
    'trabajo_de',
    'anos_de_experiencia',
    'anos_en_el_puesto_actual',
    'antiguedad_en_la_empresa_actual',
    'cuantas_personas_tenes_a_cargo',
    'tengo_edad',
    'genero', 
    'seniority',
    '_sal',
    'modalidad_de_trabajo',
    'cantidad_de_personas_en_tu_organizacion',
    'recibis_algun_tipo_de_bono',
    'contas_con_beneficios_adicionales',
    'sueldo_dolarizado']

df = df[columnas]


# Data Cleaning

mapeo = {
    "donde_estas_trabajando": "provincia", 
    "trabajo_de": "puesto",
    "cantidad_de_personas_en_tu_organizacion": "tamanio_empresa",
    "tengo_edad":"edad"
}

df = df.rename(columns=mapeo)


#  Target

media = df["_sal"].mean()
desv_std = df["_sal"].std()
LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std
df = df[(df["_sal"] >= LI_DS) & (df["_sal"] <= LS_DS)]

# Logarithmic transformation
# After removing outliers using the interquartile range method (LI_DS and LS_DS), the distribution of the salary variable _sal shows marked positive asymmetry. Therefore, in order to obtain a more suitable distribution for statistical modeling and improve the performance of predictive algorithms, logarithmic transformation (np.log) is applied.
df['target'] = np.log1p(df['_sal'])

# Numeric Variables

# Years of experience

media = df["anos_de_experiencia"].mean()
desv_std = df["anos_de_experiencia"].std()
LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std
df = df[(df["anos_de_experiencia"] >= LI_DS) & (df["anos_de_experiencia"] <= LS_DS)]

# Age

media = df["edad"].mean()
desv_std = df["edad"].std()
LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std
df = df[(df["edad"] >= LI_DS) & (df["edad"] <= LS_DS)]

# Years in the actual job

media = df["anos_en_el_puesto_actual"].mean()
desv_std = df["anos_en_el_puesto_actual"].std()
LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std
df = df[(df["anos_en_el_puesto_actual"] >= LI_DS) & (df["anos_en_el_puesto_actual"] <= LS_DS)]

# Dependents

media = df["cuantas_personas_tenes_a_cargo"].mean()
desv_std = df["cuantas_personas_tenes_a_cargo"].std()
LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std
df = df[(df["cuantas_personas_tenes_a_cargo"] >= LI_DS) & (df["cuantas_personas_tenes_a_cargo"] <= LS_DS)]

# Categorical Variables

# Province

map_provincia_region = {
    # Región Pampeana
    "Buenos Aires": "Pampeana",
    "Ciudad Autónoma de Buenos Aires": "Pampeana",
    "Santa Fe": "Pampeana",
    "Córdoba": "Pampeana",
    "Entre Ríos": "Pampeana",
    "La Pampa": "Pampeana",

    # NOA
    "Jujuy": "NOA",
    "Salta": "NOA",
    "Tucumán": "NOA",
    "Catamarca": "NOA",
    "Santiago del Estero": "NOA",
    "La Rioja": "NOA",

    # NEA
    "Misiones": "NEA",
    "Corrientes": "NEA",
    "Chaco": "NEA",
    "Formosa": "NEA",

    # Cuyo
    "Mendoza": "Cuyo",
    "San Juan": "Cuyo",
    "San Luis": "Cuyo",

    # Patagonia
    "Neuquén": "Patagonia",
    "Río Negro": "Patagonia",
    "Chubut": "Patagonia",
    "Santa Cruz": "Patagonia",
    "Tierra del Fuego": "Patagonia"
}

df["region"] = df["provincia"].map(map_provincia_region)

# Type of contract

def agrupar_contrato(tipo):
    if tipo == "Staff (planta permanente)":
        return "Staff"
    elif tipo == "Contractor":
        return "Contractor"
    elif tipo in ["Tercerizado (trabajo a través de consultora o agencia)", 
                  "Freelance"]:
        return "NoStaff_Externo"  # Agrupar tercerizados y freelance
    else:  # Cooperativa
        return "Otros"  # O eliminar si son muy pocos

df["tipo_contrato_agrupado"] = df["tipo_de_contrato"].apply(agrupar_contrato)
df = df[df["tipo_de_contrato"] != "Participación societaria en una cooperativa"]

# Job

puestos_principales = [
    "Developer",
    "SysAdmin / DevOps / SRE",
    "Manager / Director",
    "Technical Leader",
    "BI Analyst / Data Analyst",
    "QA / Tester",
    "Data Engineer",
    "Data Scientist",
    "Architect",
    "UX Designer",
    "Infosec",
    "Business Analyst",
    "Recruiter",
    "Consultant",
    "HelpDesk",
    "Networking",
    "Designer",
    "Functional Analyst",
    "Sales / Pre-Sales",
    "Scrum Master",
    "DBA (Database Administrator)"]

df["puesto_agrupado"] = df["puesto"].where(
    df["puesto"].isin(puestos_principales),
    "Otros"
)

# Gender

genero_grupo = [
    "Hombre Cis",
    "Mujer Cis"]

df["genero_agrupado"] = df["genero"].where(
    df["genero"].isin(genero_grupo),
    "Otros"
)

# New Features

df['exp_por_edad'] = df['anos_de_experiencia'] / df['edad']
df['ratio_cargo'] = df['cuantas_personas_tenes_a_cargo'] / (df['anos_de_experiencia'] + 1)
df['movilidad'] = df['anos_en_el_puesto_actual'] / (df['antiguedad_en_la_empresa_actual'] + 1)

# Variables to consider

df = df[['dedicacion', 'tipo_contrato_agrupado','anos_de_experiencia', 'seniority', 'target','modalidad_de_trabajo',
         'tamanio_empresa','region', 'genero_agrupado', 'puesto_agrupado', 'recibis_algun_tipo_de_bono', 'sueldo_dolarizado',
        'exp_por_edad', 'ratio_cargo', 'movilidad' ]]

# Split Dataset

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.target.values
y_test = df_test.target.values

del df_full_train['target']
del df_test['target']

# Model

def train(df_train, y_train):
    dicts_full_train = df_full_train.to_dict(orient='records')

    dv = DictVectorizer(sparse=False)
    X_full_train = dv.fit_transform(dicts_full_train)

    feature_names = list(dv.get_feature_names_out())

    dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                        feature_names=feature_names)

    xgb_params = {
        'eta': 0.1,
        'max_depth': 5, 
        'min_child_weight': 30, 

        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',

        'nthread': 8,
        'seed': 1,
        'verbosity': 1,
    }

    model = xgb.train(xgb_params,dfulltrain, num_boost_round=100)

    return dv, model

dv, model = train(df_full_train, y_full_train)

output_file = "model_xgb.bin"

with open (output_file, 'wb') as f_out:
    pickle.dump((dv,model),f_out)




