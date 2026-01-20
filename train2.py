#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd ## to import the dataset and analyze it
import matplotlib.pyplot as plt ## for statistical graphs
import seaborn as sns ## for statistical graphs
import numpy as np ## for work with matrices
from sklearn.model_selection import train_test_split ## for split the dataset
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle


# # Reading the dataset

# In[2]:


url = "https://raw.githubusercontent.com/caro6852lq/MachineLearning_PredictSalary/main/Data/2025.2%20-%20Sysarmy%20-%20Encuesta%20de%20remuneraci%C3%B3n%20salarial%20Argentina%20-%20Sysarmy%20-%20sueldos%20-%202025.02CLEAN.csv"


# In[3]:


df = pd.read_csv(url, skiprows=9) ## The file has header and rows that do not contain data


# In[4]:


# Selected columns
columnas = ['donde_estas_trabajando', 
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
        '_sal','modalidad_de_trabajo',
'cantidad_de_personas_en_tu_organizacion',
           'recibis_algun_tipo_de_bono',
           'contas_con_beneficios_adicionales',
           'sueldo_dolarizado']


# In[5]:


df = df[columnas]


# # Data Cleaning

# In[6]:


mapeo = {
    "donde_estas_trabajando": "provincia", 
    "trabajo_de": "puesto",
    "cantidad_de_personas_en_tu_organizacion": "tamanio_empresa",
    "tengo_edad":"edad"
}


# In[7]:


df = df.rename(columns=mapeo)


# ## Cleaning Outliers Target

# ### Method SD

# In[8]:


media = df["_sal"].mean()
desv_std = df["_sal"].std()
print(media, desv_std)


# In[9]:


LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std
print(LI_DS, LS_DS)


# In[10]:


df = df[(df["_sal"] >= LI_DS) & (df["_sal"] <= LS_DS)]
df.shape


# ### Logarithmic transformation

# After removing outliers using the interquartile range method (LI_DS and LS_DS), the distribution of the salary variable _sal shows marked positive asymmetry. Therefore, in order to obtain a more suitable distribution for statistical modeling and improve the performance of predictive algorithms, logarithmic transformation (np.log) is applied.

# In[11]:


df['target'] = np.log1p(df['_sal'])


# ## Numeric Variables

# ### Years of experience

# In[12]:


media = df["anos_de_experiencia"].mean()
desv_std = df["anos_de_experiencia"].std()

LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std

Q_a_limpiar = (df[df['anos_de_experiencia']>LS_DS]).shape

print(LI_DS, LS_DS, Q_a_limpiar )


# In[13]:


df = df[(df["anos_de_experiencia"] >= LI_DS) & (df["anos_de_experiencia"] <= LS_DS)]


# ### Age

# In[14]:


media = df["edad"].mean()
desv_std = df["edad"].std()

LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std

Q_a_limpiar = (df[df['edad']>LS_DS]).shape

print(LI_DS, LS_DS, Q_a_limpiar )


# In[15]:


df = df[(df["edad"] >= LI_DS) & (df["edad"] <= LS_DS)]


# ### Years in the actual job

# In[16]:


media = df["anos_en_el_puesto_actual"].mean()
desv_std = df["anos_en_el_puesto_actual"].std()

LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std

Q_a_limpiar = (df[df['anos_en_el_puesto_actual']>LS_DS]).shape

print(LI_DS, LS_DS, Q_a_limpiar )


# In[17]:


df = df[(df["anos_en_el_puesto_actual"] >= LI_DS) & (df["anos_en_el_puesto_actual"] <= LS_DS)]


# ### Dependents

# In[18]:


media = df["cuantas_personas_tenes_a_cargo"].mean()
desv_std = df["cuantas_personas_tenes_a_cargo"].std()

LI_DS = media - 3*desv_std
LS_DS =  media + 3*desv_std

Q_a_limpiar = (df[df['cuantas_personas_tenes_a_cargo']>LS_DS]).shape

print(LI_DS, LS_DS, Q_a_limpiar )


# In[19]:


df = df[(df["cuantas_personas_tenes_a_cargo"] >= LI_DS) & (df["cuantas_personas_tenes_a_cargo"] <= LS_DS)]
df.shape


# ## Categorical Variables

# ### Province

# In[20]:


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


# In[21]:


df["region"] = df["provincia"].map(map_provincia_region)


# ### Type of contract

# In[22]:


# Crear categorías más robustas
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


# In[23]:


df = df[df["tipo_de_contrato"] != "Participación societaria en una cooperativa"]


# ### Job

# In[24]:


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


# ### Gender

# In[25]:


genero_grupo = [
    "Hombre Cis",
    "Mujer Cis"]

df["genero_agrupado"] = df["genero"].where(
    df["genero"].isin(genero_grupo),
    "Otros"
)


# ## New Features

# In[26]:


# Crear nuevas features
df['exp_por_edad'] = df['anos_de_experiencia'] / df['edad']
df['ratio_cargo'] = df['cuantas_personas_tenes_a_cargo'] / (df['anos_de_experiencia'] + 1)
df['movilidad'] = df['anos_en_el_puesto_actual'] / (df['antiguedad_en_la_empresa_actual'] + 1)


# ## Variables to consider

# In[27]:


df = df[['dedicacion', 'tipo_contrato_agrupado','anos_de_experiencia', 'seniority', 'target','modalidad_de_trabajo',
         'tamanio_empresa','region', 'genero_agrupado', 'puesto_agrupado', 'recibis_algun_tipo_de_bono', 'sueldo_dolarizado',
        'exp_por_edad', 'ratio_cargo', 'movilidad' ]]


# # Split Dataset

# In[28]:


df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)

df_full_train = df_full_train.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_full_train = df_full_train.target.values
y_test = df_test.target.values

del df_full_train['target']
del df_test['target']


# # Model

# In[29]:


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


# In[30]:


dv, model = train(df_full_train, y_full_train)


# In[31]:


output_file = "model_xgb.bin"


# In[32]:


with open (output_file, 'wb') as f_out:
    pickle.dump((dv,model),f_out)


# In[33]:


dicts_full_train = df_full_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_full_train = dv.fit_transform(dicts_full_train)

dicts_test = df_test.to_dict(orient='records')
X_test = dv.transform(dicts_test)

feature_names = list(dv.get_feature_names_out())

dfulltrain = xgb.DMatrix(X_full_train, label=y_full_train,
                        feature_names=feature_names)


dtest = xgb.DMatrix(X_test, feature_names=feature_names)

y_pred = model.predict(dtest)

#Mido el modelo
r2_score(y_test, y_pred)


# In[39]:


example = {
    'dedicacion': 'Full-Time',
     'tipo_contrato_agrupado': 'Staff',
     'anos_de_experiencia': 3,
     'seniority': 'Semi-Senior',
     'modalidad_de_trabajo': 'Híbrido (presencial y remoto)',
     'tamanio_empresa': 'De 2001a 5000 personas',
     'region': 'Pampeana',
     'genero_agrupado': 'Mujer Cis',
    'puesto_agrupado': 'QA / Tester',
    'recibis_algun_tipo_de_bono': 'No',
    'sueldo_dolarizado': 'True',
    'exp_por_edad': 0.07,
    'ratio_cargo': 0,
    'movilidad':0
}


# In[35]:


df.head(1)


# In[40]:


X = dv.transform([example])


# In[41]:


d = xgb.DMatrix(X, feature_names=feature_names)


# In[48]:


suggested_salary_direct = model.predict(d)


# In[49]:


suggested_salary_direct


# In[50]:


suggested_salary = np.expm1(model.predict(d)).astype(int)


# In[47]:


suggested_salary


# In[51]:


np.expm1(14.736263).astype(int)


# In[58]:


CVexample = {
    'dedicacion': 'Full-Time',
     'tipo_contrato_agrupado': 'Staff',
     'anos_de_experiencia': 5,
     'seniority': 'Senior',
     'modalidad_de_trabajo': '100% remoto',
     'tamanio_empresa': 'De 101 a 200 personas',
     'region': 'NOA',
     'genero_agrupado': 'Mujer Cis',
    'puesto_agrupado': 'BI Analyst / Data Analyst ',
    'recibis_algun_tipo_de_bono': 'No',
    'sueldo_dolarizado': 'False',
    'exp_por_edad': 0.0,
    'ratio_cargo': 0,
    'movilidad':0
}


# In[59]:


X = dv.transform([CVexample])
d = xgb.DMatrix(X, feature_names=feature_names)
suggested_salary = np.expm1(model.predict(d)).astype(int)
suggested_salary


# In[68]:


CVNexample = {
    'dedicacion': 'Full-Time',
     'tipo_contrato_agrupado': 'Staff',
     'anos_de_experiencia': 2,
     'seniority': 'Junior',
     'modalidad_de_trabajo': '100% remoto',
     'tamanio_empresa': 'De 101 a 200 personas',
     'region': 'NOA',
     'genero_agrupado': 'Mujer Cis',
    'puesto_agrupado': 'Data Scientist',
    'recibis_algun_tipo_de_bono': 'No',
    'sueldo_dolarizado': 'True',
    'exp_por_edad': 0.0,
    'ratio_cargo': 0,
    'movilidad':0
}


# In[69]:


X = dv.transform([CVNexample])
d = xgb.DMatrix(X, feature_names=feature_names)
suggested_salary = np.expm1(model.predict(d)).astype(int)
suggested_salary


# In[ ]:




