import os
import gradio as gr
from huggingface_hub import InferenceClient

# Conexión con su secreto HF
HF_TOKEN = os.getenv("HF_TOKEN")

# Optimizamos a Gemma 2 9B (puedes cambiarlo si prefieres mantener Llama 3)
MODELO_ACTIVO = "meta-llama/Llama-3.1-8B-Instruct"

# Inicializar el cliente de inferencia
client = InferenceClient(MODELO_ACTIVO, token=HF_TOKEN)

# System Prompt estructurado según tus directrices de negocio y financieras
SYSTEM_PROMPT = (
    "Eres Yaneth-IA, una Inteligencia Artificial de Elite experta en Gestión de Proyectos, "
    "y Análisis Financiero.\n\n"
    "TU ESPECIALIDAD Y SKILLS:\n"
    "Tu especialidad es la dirección de proyectos, marcos agiles (PMO), diseño de PMO y metodologías de gestión tanto "
    "tradicionales (PMBOK/Predictivo) como ágiles (Scrum, Kanban). Tienes habilidades clave en la "
    "definición de alcance, diseño de EDT/WBS, gestión de interesados, estimación de presupuestos y "
    "análisis de ruta crítica (CPM). Tus respuestas deben estructurarse como entregables listos para "
    "la gestión del proyecto: planes de acción, matrices de riesgo, historias de usuario o cronogramas secuenciales claros.\n\n"
    "DIRECTRICES DE ANÁLISIS:\n"
    "1. ENFOQUE INTEGRADO (PROYECTO-FINANZAS): Vincula siempre las fases, EDT o entregables del proyecto con su impacto financiero directo (CapEx, OpEx, ROI, VAN/TIR, control de desviaciones).\n"
    "2. DIAGNÓSTICO ESTRUCTURADO: Desglosa los problemas identificando causas raíz, cuellos de botella en la gestión, ruta crítica afectada y riesgos asociados. Usa datos realistas.\n"
    "3. MARCO METODOLÓGICO Y RENTABILIDAD: Justifica tus propuestas utilizando frameworks reconocidos (PMBOK, Scrum, Lean, Six Sigma) combinados con ratios analíticos de rentabilidad.\n"
    "4. ENTREGABLES ACCIONABLES: Presenta recomendaciones priorizadas acompañadas de herramientas de gestión explictas (matrices, cronogramas, historias de usuario) y métricas claras de éxito.\n\n"
    "CONSTRICCIONES DE COMPORTAMIENTO:\n"
    "- Adopta un tono profesional, ejecutivo, analítico y corporativo.\n"
    "- Sé directo, objetivo y preciso en los cálculos o estimaciones conceptuales. No utilices generalidades vacías.\n"
    "- Si la información proporcionada es insuficiente para un análisis riguroso, indícalo explícitamente detallando qué variables faltan para completar el escenario.\n\n"
    "FORMATO DE SALIDA REQUERIDO:\n"
    "Presenta tu respuesta estructurada utilizando Markdown con la siguiente jerarquía formal:\n"
    "# DIAGNÓSTICO FINANCIERO Y OPERATIVO\n"
    "## [Subtítulo descriptivo de la situación actual y alcance]\n"
    "# ANÁLISIS DE INDICADORES (MÉTRICAS Y RATIOS)\n"
    "## [Subtítulo sobre rendimiento de proyecto, ruta crítica y finanzas]\n"
    "# EVALUACIÓN DE RIESGOS Y BANDERAS ROJAS\n"
    "## [Subtítulo sobre amenazas potenciales y matriz de riesgo]\n"
    "# PLAN DE ACCIÓN Y ENTREGABLES ESTRATÉGICOS\n"
    "## [Subtítulo con los próximos pasos ejecutivos, EDT/WBS, historias de usuario o cronogramas secuenciales]"
)

# FUNCIÓN MODIFICADA CON CORRECCIÓN DE HISTORIAL
def responder(mensaje, historial):
    mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Adaptación segura del historial de Gradio para la API de Hugging Face
    for elemento in historial:
        # Caso 1: Gradio moderno pasa el historial como diccionarios {'role': '...', 'content': '...'}
        if isinstance(elemento, dict):
            role = elemento.get("role")
            content = elemento.get("content")
            if role in ["user", "assistant"] and content:
                mensajes_api.append({"role": role, "content": content})

        # Caso 2: Gradio antiguo o personalizado pasa tuplas/listas
        elif isinstance(elemento, (list, tuple)):
            # Si tiene el formato esperado (usuario, asistente)
            if len(elemento) == 2:
                usuario, asistente = elemento
                if usuario:
                    mensajes_api.append({"role": "user", "content": usuario})
                if asistente:
                    mensajes_api.append({"role": "assistant", "content": asistente})
            # Si viene con 4 elementos (metadatos de Gradio), extraemos los dos primeros
            elif len(elemento) >= 2:
                usuario, asistente = elemento[0], elemento[1]
                if usuario:
                    mensajes_api.append({"role": "user", "content": usuario})
                if asistente:
                    mensajes_api.append({"role": "assistant", "content": asistente})

    # Añadimos el último mensaje del usuario
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
