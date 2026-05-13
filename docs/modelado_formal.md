# Modelado Formal: Haikus Técnicos

El problema consiste en la generación automatizada de mensajes poéticos que aplican conceptos de tecnología respetando estrictas reglas de métrica y estructura.

## Restricciones Estructurales (Hard Constraints)
Para que un texto generado sea considerado como válido (éxito), debe cumplir con las siguientes condiciones lógicas $C$:

1. **Estructura de líneas ( $C_1$ )**: El texto debe contener exactamente tres (3) líneas no vacías.
2. **Métrica ( $C_2$ )**: 
   - La primera línea debe tener exactamente 5 sílabas.
   - La segunda línea debe tener exactamente 7 sílabas.
   - La tercera línea debe tener exactamente 5 sílabas.
3. **Inclusión de contenido ( $C_3$ )**: El mensaje debe contener obligatoriamente una palabra clave dada $w$ en cualquier posición del texto.

## Restricciones Semánticas (Soft Constraints)
- **Dominio y Tono**: El poema debe hablar de un tema de software o tecnología especificado en el input. Dado que estas condiciones son subjetivas, requieren evaluación cualitativa o el uso de un LLM-as-a-judge para validarlas.

## Input del Sistema
Un vector de entrada $X = [t, w]$, donde:
- $t$: es el tema técnico especificado (ej. "Machine Learning").
- $w$: es una palabra que se exige incluir (ej. "datos").

## Output Esperado
Un texto $Y$ tal que eval$(Y) \to \{0,1\}$, siendo $1$ si cumple $C_1 \land C_2 \land C_3$, y $0$ de lo contrario.
