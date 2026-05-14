# Informe Técnico Final: Generación de Textos bajo Restricciones Estructurales (CSP Dinámico)

## 1. Introducción
El objetivo de este proyecto universitario es la construcción de un sistema capaz de delegar la generación de mensajes a Modelos de Lenguaje (LLMs) imponiendo restricciones paramétricas variadas, tales como "mínimo de palabras", "número exacto de líneas", "formato de salida" y restricciones semánticas sutiles como "tono".

A diferencia de los enfoques convencionales donde se busca modelar el lenguaje mediante ajustes finos de los pesos neuronales, este proyecto aborda el dominio desde la perspectiva de la Ingeniería de Prompts (Prompt Engineering) y Sistemas de Agentes.

## 2. Metodología y Arquitectura
El sistema se compone de dos componentes esenciales, orquestados en un ciclo iterativo. Empleamos modelos locales montados sobre **Ollama**: Llama 3 (8B), Mistral (7B) y Phi-3 (3.8B).
- **El Generador (`DynamicGenerator`)**: Ensambla prompts estructurados para indicar explícitamente las reglas. Integra además una función de *Reflection Loop* (Autocorrección). 
- **El Evaluador (`DynamicEvaluator`)**: Evalúa el cumplimiento (0/1) de cada regla. Las reglas formales (como longitudes o parsing de JSON) están implementadas algorítmicamente mediante Regex, pero las reglas semánticas (tono) utilizan el paradigma **LLM-as-a-Judge**, en este caso usando internamente un Llama 3 configurado con `temperature=0.0`.

## 3. Experimentos y Variantes Evaluadas
### Variante A: Prompt Base (Zero-shot Directo)
El LLM recibe las reglas en texto libre. Descubrimos que modelos ligeros padecen el *Chatty Assistant Syndrome*, añadiendo texto inicial/final (ej. "Aquí tiene su JSON:") lo que destrozaba la adherencia a la regla "formato puro".

### Variante B: Prompt Estructurado (Marcadores XML)
Empleamos separadores tipo `<system>`, `<constraints>` y pautas duras contra la cháchara. Mejoró significativamente la robustez del parseo, pero en presencia de *Restricciones Blandas* complejas (ej. "Escribe un poema de enojo divertido"), los modelos perdían la capacidad fundamental de contar líneas u obligatoriedad de palabras, un fenómeno conocido como *Catastrophic Interference*.

### Variante C: Bucle de Reflexión (Auto-corrección Iterativa)
Esta es la implementación final y definitiva del proyecto. El script en Python corrobora rigurosamente las restricciones *Hard*. Si Llama 3, Mistral o Phi-3 erran, Python les re-inyecta sus fallos diciendo: "El texto es muy corto" o "Es obligatorio usar X palabra", exigiéndoles reescribir. 

## 4. Resultados Analíticos (Comprensión de Costes)
La auto-corrección fue abrumadoramente exitosa en **Llama 3**. Gracias a que Python indicó dinámicamente sus fallos, Llama 3 restauró su *Formatting Compliance* del formato al 100%.

Sin embargo, analizamos un *trade-off* importante: el **Costo Computacional vs Exactitud de Modelos**.
Nuestro script ahora registra tiempos de inferencia:
- Los modelos logran acierto directo pero la generación cuesta $T_{inicial}$.
- Cuando un modelo pequeño (como Mistral o Phi-3) falla e ingresa al ciclo de auto-corrección (Retries), el tiempo se dobla $T_{retry}$, y en los casos empíricos de modelos de < 7B parámetros, corregir un aspecto (como el número de líneas) rompe lo semántico.
Por tanto, el abordaje mediante Bucles de Reflexión en LLMs en entornos locales tiene su frontera teórica en el umbral cognitivo natural del número de parámetros del modelo base.

## 5. Conclusión
El sistema propuesto no solo resuelve el Control de Formato y Estructura dinámica, sino que aporta métricas tangibles (Accuracy, Lexical Compliance, Latencia) en entornos multivariables para IAs generativas de propósito general. Llama 3 validó que, siempre y cuando el tamaño del modelo soporte tareas de edición compleja (*editing* vs *generating*), un sistema modular con evaluadores duros (RegEx) acoplados perimetralmente alrededor del LLM, puede suplir totalmente las falencias estructurales de base de las Inteligencias Artificiales modernas.
