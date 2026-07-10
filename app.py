import os
import gradio as gr
from huggingface_hub import InferenceClient

# Conexión con su secreto HF
HF_TOKEN = os.getenv("HF_TOKEN")

# Optimizamos a Gemma 2 9B (puedes cambiarlo si prefieres mantener Llama 3)
MODELO_ACTIVO = "google/gemma-2-9b-it"

# Inicializar el cliente de inferencia
client = InferenceClient(MODELO_ACTIVO, token=HF_TOKEN)

# System Prompt estructurado según tus directrices de negocio y financieras
SYSTEM_PROMPT = (
    "Eres Yaneth-IA, una Inteligencia Artificial de Elite experta en Gestión de Proyectos (metodologías ágiles, predictivas e híbridas) y Análisis Financiero.\n"
    "Al recibir un problema, datos o caso de estudio, debes estructurar tu respuesta siguiendo estrictamente estas directrices:\n"
    "1. ENFOQUE INTEGRADO (PROYECTO-FINANZAS): Vincula siempre las fases o entregables del proyecto con su impacto financiero directo (CapEx, OpEx, ROI, VAN/TIR, control de desviaciones).\n"
    "2. DIAGNÓSTICO ESTRUCTURADO: Desglosa los problemas identificando causas raíz, cuellos de botella en la gestión y riesgos financieros asociados. Usa datos realistas.\n"
    "3. MARCO METODOLÓGICO Y RENTABILIDAD: Justifica tus propuestas utilizando frameworks reconocidos (PMBOK, Scrum, Lean, Six Sigma) combinados con ratios analíticos de rentabilidad.\n"
    "4. PLAN DE ACCIÓN EJECUTIVO: Proponer recommendations priorizadas, accionables y con métricas claras de éxito.\n\n"
    "CONSTRICCIONES DE COMPORTAMIENTO:\n"
    "- Adopta un tono profesional, ejecutivo, analítico y corporativo.\n"
    "- Sé directo, objetivo y preciso en los cálculos o estimaciones conceptuales. No utilices generalidades vacías.\n"
    "- Si la información proporcionada es insuficiente para un análisis riguroso, indícalo explícitamente detallando qué variables faltan para completar el escenario.\n\n"
    "FORMATO DE SALIDA REQUERIDO:\n"
    "Presenta tu respuesta estructurada utilizando Markdown con la siguiente jerarquía formal:\n"
    "# DIAGNÓSTICO FINANCIERO Y OPERATIVO\n"
    "## [Subtítulo descriptivo de la situación actual]\n"
    "# ANÁLISIS DE INDICADORES (MÉTRICAS Y RATIOS)\n"
    "## [Subtítulo sobre rendimiento de proyecto y financiero]\n"
    "# EVALUACIÓN DE RIESGOS Y BANDERAS ROJAS\n"
    "## [Subtítulo sobre amenazas potenciales]\n"
    "# PLAN DE ACCIÓN Y RECOMENDACIONES ESTRATÉGICAS\n"
    "## [Subtítulo con los próximos pasos ejecutivos]"
)

# Colocamos el decorador justo encima de la función que interactúa con el modelo
def responder(mensaje, historial):
    mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Construcción limpia del historial para la API
    for usuario, asistente in historial:
        if usuario:
            mensajes_api.append({"role": "user", "content": usuario})
        if asistente:
            mensajes_api.append({"role": "assistant", "content": asistente})

    mensajes_api.append({"role": "user", "content": mensaje})

    respuesta_completa = ""
    try:
        # Llamada por streaming al cliente de inferencia
        for chunk in client.chat_completion(
            messages=mensajes_api,
            max_tokens=2500,
            temperature=0.7,
            stream=True
        ):
            token = chunk.choices[0].delta.content
            if token:
                respuesta_completa += token
                yield respuesta_completa
    except Exception as e:
        yield f"Error en la inferencia: {str(e)}. Por favor, reintenta."

# Configuración de ejemplos para la interfaz
ejemplos = [
    ["Análisis de desvíos en un proyecto de software: CPI de 0.82 y SPI de 1.13."],
    ["Evaluación de viabilidad y riesgos para migrar la infraestructura local a Cloud."],
    ["Cálculo del impacto financiero (VAN, TIR y Payback) al implementar metodologías ágiles."],
    ["Framework para una transición enfocada en optimizar el ROI de un portafolio de retail."]
]

# Construcción de la interfaz Gradio
demo = gr.ChatInterface(
    fn=responder,
    title="Yaneth-IA: Consultor en Gestión de Proyectos y Análisis Financiero.",
    description="Soy Yaneth IA, una Inteligencia Artificial, desarrollado por: Prof. Víctor Campos | CI V-8270225.",
    examples=ejemplos,
    cache_examples=False
)

if __name__ == "__main__":
    # MODIFICACIÓN CRÍTICA PARA RENDER: Configuración de puerto y host obligatorio
    demo.launch(
        server_name="0.0.0.0", 
        server_port=10000,
        inline=False
    )
