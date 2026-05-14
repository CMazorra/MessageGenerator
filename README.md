# MessageGenerator: Constraint Satisfaction via LLMs

Este repositorio contiene todo el entorno de desarrollo, evaluación y reportes para el proyecto universitario "Generation of messages under structural constraints".

## Resumen del Proyecto
El sistema plantea un **Dynamic Constraint Satisfaction Problem (CSP)** en el cual diversos LLMs locales (Llama 3, Mistral, Phi-3) generan textos mientras intentan satisfacer restricciones estructurales duras (conteo de palabras, líneas exactas, llaves de JSON) y restricciones semánticas o "suaves" (tono). El sistema incluye:
- **LLM-as-a-Judge**: Un modelo secundario evalúa autónomamente si el tono se cumple.
- **Reflection Loop (Auto-corrección)**: Un módulo que detecta los errores y re-prompta al modelo en un bucle cerrado.

## Requisitos Previos
- **Docker** y **Docker Compose** instalados o **Ollama** independiente.
- **Python 3.11** o superior (si se desea ejecutar scripts localmente fuera de docker).

## Despliegue con Docker (Recomendado)

En la raíz del proyecto, ejecuta:
```bash
docker-compose up --build
```
Esto levantará dos contenedores:
1. `mgen_ollama`: Inicia el servidor de inferencia y extrae los modelos `llama3`, `mistral` y `phi3`.
2. `mgen_app`: Es el contenedor de Python donde ejecutarás tus scripts.

## Uso del Entorno

Si quieres correr el experimento principal (que procesará el dataset y arrojará las métricas) debes ingresar al contenedor Python:

```bash
docker exec -it <nombre_del_contenedor_mgen_app> bash
```
Y luego ejecutar:
```bash
python src/main.py
```
*Nota*: Requerirá tiempo para evaluar todos los casos especialmente si tu ordenador usa CPU para la inferencia.

### Pruebas Unitarias
Para asegurar la fiabilidad de las validaciones en python (las restricciones duras), utilizamos pytest:
```bash
python -m pytest tests/test_evaluator.py
```

## Estructura del Proyecto

- `data/` \- Dataset en JSON y tabla de resultados CSV.
- `src/` \- Código principal (`generator.py`, `evaluator.py`, `main.py`).
- `docs/` \- Documentación, informe técnico y análisis de resultados de cada variante.
- `tests/` \- Pruebas unitarias automatizadas. 
