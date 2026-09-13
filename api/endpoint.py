from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
import joblib

app = FastAPI(
    title="Predicción de salario",
    description=(
        "Endpoint que recibe una oferta de trabajo (skills, industria y coste de vida "
        "de la región) y devuelve un salario estimado usando un modelo Random Forest "
        "entrenado sobre el dataset de LinkedIn Job Postings."
    ),
)

# Cargamos el modelo ya entrenado y la lista de columnas que espera,
# generados en el notebook analisis_exploratorio_dataset.ipynb
modelo = joblib.load("modelo_salario.pkl")
columnas = joblib.load("columnas_modelo.pkl")


class OfertaEntrada(BaseModel):
    """Datos de una oferta de trabajo, tal como los enviaría quien use el endpoint."""
    skills: list[str] = Field(
        ..., description="Lista de skills mencionadas en la oferta, ej. ['Python', 'SQL']"
    )
    industria: str = Field(
        ..., description="Industria de la oferta, ej. 'Software Development'"
    )
    indice_coste_vida: float = Field(
        ..., description="Índice de coste de vida del estado donde está la oferta"
    )


@app.get("/")
def estado():
    """Comprueba que el endpoint está activo."""
    return {"status": "ok", "mensaje": "Endpoint de predicción de salario activo"}


@app.post("/predecir")
def predecir_salario(oferta: OfertaEntrada):
    """
    Predice el salario de una oferta a partir de sus skills, industria
    e índice de coste de vida, reproduciendo la lógica de la función
    predecir_salario() del notebook de exploración.
    """
    # Fila vacía con todas las columnas que espera el modelo, a 0
    fila = pd.Series(0, index=columnas, dtype=float)

    # Marcamos a 1 las skills que aparecen en la oferta
    for skill in oferta.skills:
        if skill in fila.index:
            fila[skill] = 1

    # Marcamos la industria correspondiente
    columna_industria = f"industria_{oferta.industria}"
    if columna_industria in fila.index:
        fila[columna_industria] = 1

    # Coste de vida de la región
    fila["indice_coste_vida"] = oferta.indice_coste_vida

    # Predicción con el modelo entrenado
    salario_estimado = modelo.predict(pd.DataFrame([fila]))[0]
    return {"salario_estimado": round(float(salario_estimado), 2)}