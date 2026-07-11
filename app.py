import os
import gradio as gr
from huggingface_hub import InferenceClient
from weasyprint import HTML

# Conexión con su secreto HF
HF_TOKEN = os.getenv("HF_TOKEN")

# Optimizamos a Gemma 2 9B (puedes cambiarlo si prefieres mantener Llama 3)
MODELO_ACTIVO = "Qwen/Qwen2.5-72B-Instruct"

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
    "4. ENTREGABLES ACCIONABLES: Presenta recomendaciones priorizadas acompañadas de herramientas de gestión explícitas (matrices, cronogramas, historias de usuario) y métricas claras de éxito.\n\n"
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

def procesar_historial_api(historial):
    mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}]
    for elemento in historial:
        # Caso 1: Gradio moderno pasa el historial como diccionarios
        if isinstance(elemento, dict):
            role = elemento.get("role")
            content = elemento.get("content")
            if role in ["user", "assistant"] and content:
                mensajes_api.append({"role": role, "content": content})
        
        # Caso 2: Gradio antiguo o personalizado pasa tuplas/listas
        elif isinstance(elemento, (list, tuple)) and len(elemento) >= 2:
            usuario, asistente = elemento[0], elemento[1]
            if usuario: 
                mensajes_api.append({"role": "user", "content": usuario})
            if asistente: 
                mensajes_api.append({"role": "assistant", "content": asistente})
    return mensajes_api

def responder(mensaje, historial):
    if not mensaje.strip():
        yield historial
        return

    mensajes_api = procesar_historial_api(historial)
    mensajes_api.append({"role": "user", "content": mensaje})

    # El chatbot de Gradio moderno requiere actualizar agregando el mensaje del usuario primero
    historial.append({"role": "user", "content": mensaje})
    yield historial

    respuesta_completa = ""
    try:
        for chunk in client.chat_completion(
            messages=mensajes_api,
            max_tokens=2500,
            temperature=0.7,
            stream=True
        ):
            token = chunk.choices[0].delta.content
            if token:
                respuesta_completa += token
                # Actualizamos o creamos la respuesta del asistente en el historial
                if historial[-1]["role"] == "assistant":
                    historial[-1]["content"] = respuesta_completa
                else:
                    historial.append({"role": "assistant", "content": respuesta_completa})
                yield historial
    except Exception as e:
        error_msg = f"Error en la inferencia: {str(e)}. Por favor, reintenta."
        if historial[-1]["role"] == "assistant":
            historial[-1]["content"] = error_msg
        else:
            historial.append({"role": "assistant", "content": error_msg})
        yield historial

# EXPORTACIÓN A PDF CORPORATIVO CON WEASYPRINT
def exportar_a_pdf(historial):
    if not historial:
        return None
        
    import markdown
    
    # Estructura del HTML con estilos CSS profesionales aptos para impresión
    html_content = """
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @page {
                size: A4;
                margin: 20mm 15mm;
                @bottom-right {
                    content: "Página " counter(page) " de " counter(pages);
                    font-family: 'Arial', sans-serif;
                    font-size: 9pt;
                    color: #718096;
                }
                @bottom-left {
                    content: "Yaneth-IA | Reporte Ejecutivo";
                    font-family: 'Arial', sans-serif;
                    font-size: 9pt;
                    color: #718096;
                }
            }
            body {
                font-family: 'Arial', sans-serif;
                color: #2d3748;
                line-height: 1.5;
                margin: 0;
                padding: 0;
            }
            .header-banner {
                background-color: #1a365d;
                color: white;
                padding: 20px;
                margin-bottom: 25px;
                border-radius: 4px;
            }
            .header-banner h1 {
                margin: 0;
                font-size: 20pt;
                letter-spacing: 0.5px;
            }
            .header-banner p {
                margin: 5px 0 0 0;
                font-size: 10pt;
                opacity: 0.8;
            }
            .bloque-conversacion {
                margin-bottom: 20px;
                page-break-inside: avoid;
            }
            .rol-usuario {
                font-weight: bold;
                color: #2b6cb0;
                font-size: 11pt;
                margin-bottom: 5px;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 3px;
            }
            .rol-asistente {
                font-weight: bold;
                color: #2c5282;
                font-size: 11pt;
                margin-bottom: 5px;
                border-bottom: 1px solid #e2e8f0;
                padding-bottom: 3px;
            }
            .contenido {
                font-size: 10pt;
                text-align: justify;
                margin-bottom: 15px;
                padding-left: 5px;
            }
            /* Formateo de los entregables markdown de Yaneth-IA */
            h1, h2, h3 { color: #1a365d; page-break-after: avoid; }
            h1 { font-size: 14pt; border-left: 4px solid #1a365d; padding-left: 8px; margin-top: 20px; }
            h2 { font-size: 12pt; color: #4a5568; }
            table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 9pt; }
            th, td { border: 1px solid #cbd5e0; padding: 8px; text-align: left; }
            th { background-color: #f7fafc; color: #2d3748; font-weight: bold; }
            tr:nth-child(even) { background-color: #f8fafc; }
        </style>
    </head>
    <body>
        <div class="header-banner">
            <h1>MINUTA Y REPORTE CONSULTIVO | YANETH-IA</h1>
            <p>Consultoría en Gestión de Proyectos, PMO y Análisis Financiero de Inversiones</p>
        </div>
    """
    
    # Procesamos cada elemento del historial y lo convertimos a HTML estructurado
    for elemento in historial:
        user_text = ""
        bot_text = ""
        
        if isinstance(elemento, dict):
            if elemento.get("role") == "user":
                user_text = elemento.get("content", "")
            elif elemento.get("role") == "assistant":
                bot_text = elemento.get("content", "")
        elif isinstance(elemento, (list, tuple)) and len(elemento) >= 2:
            user_text, bot_text = elemento[0], elemento[1]
            
        if user_text:
            html_content += f"""
            <div class="bloque-conversacion">
                <div class="rol-usuario">▲ SOLICITUD DEL CLIENTE / USUARIO:</div>
                <div class="contenido">{markdown.markdown(user_text)}</div>
            </div>"""
        if bot_text:
            html_content += f"""
            <div class="bloque-conversacion">
                <div class="rol-asistente">◆ ENTREGABLE DE CONSULTORÍA (YANETH-IA):</div>
                <div class="contenido">{markdown.markdown(bot_text)}</div>
            </div>"""
            
    html_content += "</body></html>"
    
    # Guardar archivo PDF temporal
    pdf_path = "Reporte_Consultoria_YanethIA.pdf"
    HTML(string=html_content).write_pdf(pdf_path)
    return pdf_path

# Diseño de Interfaz Avanzada con Gradio Blocks
with gr.Blocks(title="Yaneth-IA Executive Suite") as demo:
    gr.Markdown("# Yaneth-IA: Consultor en Gestión de Proyectos y Análisis Financiero")
    gr.Markdown("Desarrollado por: Prof. Víctor Campos | CI V-8270225")
    
    chatbot = gr.Chatbot(type="messages", label="Mesa de Trabajo Consultiva")
    
    # Agrupamos la entrada de texto para que se vea ordenada
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Escribe tu consulta sobre PMO, CapEx, OpEx, EDT o riesgos aquí...", 
            label="Entrada de Consulta",
            scale=4
        )
    
    # Fila de botones de control y acción
    with gr.Row():
        btn_enviar = gr.Button("Enviar Consulta", variant="primary", scale=1)
        btn_limpiar = gr.Button("Limpiar Chat", scale=1)
    
    # Separador visual para el área de exportación
    gr.Markdown("---")
    gr.Markdown("### 📄 Opciones de Exportación Corporativa")
    
    # Colocamos el botón de PDF y el cuadro de descarga uno al lado del otro
    with gr.Row():
        btn_pdf = gr.Button("Formatear y Exportar Minuta a PDF", variant="secondary")
        archivo_descarga = gr.File(label="Descargar Reporte PDF Generado", interactive=False)
        
    # Función auxiliar para limpiar la caja de texto tras enviar el mensaje
    def limpiar_entrada():
        return ""

    # Acciones de la interfaz (Enviar con Click o con Enter)
    msg.submit(responder, [msg, chatbot], chatbot).then(limpiar_entrada, None, msg)
    btn_enviar.click(responder, [msg, chatbot], chatbot).then(limpiar_entrada, None, msg)
    
    # Acción para reiniciar el tablero
    btn_limpiar.click(lambda: [], None, chatbot, queue=False)
    
    # Evento para generar y descargar el PDF
    btn_pdf.click(exportar_a_pdf, inputs=[chatbot], outputs=[archivo_descarga])

if __name__ == "__main__":
    # CONFIGURACIÓN OBLIGATORIA PARA RENDER
    demo.launch(
        server_name="0.0.0.0", 
        server_port=10000,
        inline=False
    )