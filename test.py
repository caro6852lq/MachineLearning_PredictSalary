import requests

url = "http://localhost:9696/predict"

worker = {
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

response = requests.post(url, json=worker)
print(response.json())
