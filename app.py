import os
import gradio as gr
from huggingface_hub import InferenceClient
from weasyprint import HTML

# ═══════════════════════════════════════════════════════════════
# CONFIGURACIÓN Y CONSTANTES
# ═══════════════════════════════════════════════════════════════
HF_TOKEN = os.getenv("HF_TOKEN")
MODELO_ACTIVO = "Qwen/Qwen2.5-72B-Instruct"

client = InferenceClient(MODELO_ACTIVO, token=HF_TOKEN)

SYSTEM_PROMPT = (
    "Eres Yaneth-IA, una Inteligencia Artificial de Élite experta en Gestión de Proyectos "
    "y Análisis Financiero.\n\n"
    "TU ESPECIALIDAD Y SKILLS:\n"
    "Tu especialidad es la dirección de proyectos, marcos ágiles (PMO), diseño de PMO y metodologías de gestión tanto "
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

# ═══════════════════════════════════════════════════════════════
# FUNCIONES AUXILIARES
# ═══════════════════════════════════════════════════════════════

def procesar_historial_api(historial):
    """
    Convierte el historial de Gradio al formato de mensajes de la API de HuggingFace.
    Soporta tanto formato dict (moderno) como tuplas/listas (legado).
    """
    mensajes_api = [{"role": "system", "content": SYSTEM_PROMPT}]

    for elemento in historial:
        # Caso 1: Formato moderno de Gradio (diccionarios)
        if isinstance(elemento, dict):
            role = elemento.get("role")
            content = elemento.get("content")
            if role in ["user", "assistant"] and content:
                mensajes_api.append({"role": role, "content": content})

        # Caso 2: Formato legado de Gradio (tuplas/listas)
        elif isinstance(elemento, (list, tuple)) and len(elemento) >= 2:
            usuario, asistente = elemento[0], elemento[1]
            if usuario:
                mensajes_api.append({"role": "user", "content": usuario})
            if asistente:
                mensajes_api.append({"role": "assistant", "content": asistente})

    return mensajes_api


def responder(mensaje, historial):
    """
    Genera una respuesta en streaming usando el modelo de HuggingFace.
    Maneja errores de forma robusta y actualiza el historial progresivamente.
    """
    if not mensaje or not mensaje.strip():
        yield historial
        return

    # Preparar mensajes para la API
    mensajes_api = procesar_historial_api(historial)
    mensajes_api.append({"role": "user", "content": mensaje})

    # Agregar mensaje del usuario al historial visual
    historial = list(historial)  # Crear copia para evitar mutación inesperada
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
                # Actualizar la última entrada del asistente o crearla
                if historial and historial[-1].get("role") == "assistant":
                    historial[-1]["content"] = respuesta_completa
                else:
                    historial.append({"role": "assistant", "content": respuesta_completa})
                yield historial

    except Exception as e:
        error_msg = f"⚠️ Error en la inferencia: {str(e)}. Por favor, reintenta."
        if historial and historial[-1].get("role") == "assistant":
            historial[-1]["content"] = error_msg
        else:
            historial.append({"role": "assistant", "content": error_msg})
        yield historial


# ═══════════════════════════════════════════════════════════════
# EXPORTACIÓN A PDF — VERSIÓN CORREGIDA Y OPTIMIZADA
# ═══════════════════════════════════════════════════════════════

def exportar_a_pdf(historial):
    """
    Exporta el historial del chat a un PDF corporativo profesional.

    CORRECCIONES APLICADAS:
    - Manejo robusto de historial vacío o None
    - Conversión de Markdown a HTML con la librería 'markdown'
    - Procesamiento correcto de ambos formatos de historial (dict y tuplas)
    - CSS mejorado con soporte para tablas, listas, código y énfasis
    - Manejo de excepciones en la generación del PDF
    - Ruta de salida absoluta para evitar problemas de permisos
    """
    import markdown
    from datetime import datetime

    # ── Validaciones iniciales ──────────────────────────────────
    if not historial:
        return None

    # Filtrar solo elementos válidos
    historial_filtrado = []
    for elemento in historial:
        if isinstance(elemento, dict) and elemento.get("role") in ["user", "assistant"]:
            historial_filtrado.append(elemento)
        elif isinstance(elemento, (list, tuple)) and len(elemento) >= 2:
            historial_filtrado.append(elemento)

    if not historial_filtrado:
        return None

    # ── Construcción del HTML ───────────────────────────────────
    fecha_hora = datetime.now().strftime("%d/%m/%Y %H:%M")

    html_content = f"""<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <title>Reporte Consultivo Yaneth-IA</title>
    <style>
        @page {{
            size: A4;
            margin: 20mm 15mm 25mm 15mm;
            @bottom-right {{
                content: "Página " counter(page) " de " counter(pages);
                font-family: 'Arial', sans-serif;
                font-size: 9pt;
                color: #718096;
            }}
            @bottom-left {{
                content: "Yaneth-IA | Reporte Ejecutivo";
                font-family: 'Arial', sans-serif;
                font-size: 9pt;
                color: #718096;
            }}
        }}

        body {{
            font-family: 'Arial', 'Helvetica', sans-serif;
            color: #2d3748;
            line-height: 1.6;
            margin: 0;
            padding: 0;
            font-size: 10.5pt;
        }}

        /* ── Encabezado corporativo ── */
        .header-banner {{
            background: linear-gradient(135deg, #1a365d 0%, #2c5282 100%);
            color: white;
            padding: 24px 20px;
            margin-bottom: 30px;
            border-radius: 6px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.15);
        }}
        .header-banner h1 {{
            margin: 0 0 6px 0;
            font-size: 20pt;
            letter-spacing: 0.5px;
            font-weight: 700;
        }}
        .header-banner .subtitle {{
            margin: 0;
            font-size: 10pt;
            opacity: 0.85;
        }}
        .header-banner .meta {{
            margin: 8px 0 0 0;
            font-size: 9pt;
            opacity: 0.7;
            border-top: 1px solid rgba(255,255,255,0.2);
            padding-top: 6px;
        }}

        /* ── Bloques de conversación ── */
        .bloque-conversacion {{
            margin-bottom: 24px;
            page-break-inside: avoid;
        }}

        .rol-usuario {{
            font-weight: 700;
            color: #2b6cb0;
            font-size: 11pt;
            margin-bottom: 6px;
            border-bottom: 2px solid #bee3f8;
            padding-bottom: 4px;
            display: flex;
            align-items: center;
        }}
        .rol-usuario::before {{
            content: "▲";
            margin-right: 8px;
            font-size: 10pt;
        }}

        .rol-asistente {{
            font-weight: 700;
            color: #276749;
            font-size: 11pt;
            margin-bottom: 6px;
            border-bottom: 2px solid #c6f6d5;
            padding-bottom: 4px;
            display: flex;
            align-items: center;
        }}
        .rol-asistente::before {{
            content: "◆";
            margin-right: 8px;
            font-size: 10pt;
        }}

        .contenido {{
            text-align: justify;
            padding-left: 8px;
            color: #2d3748;
        }}

        /* ── Markdown renderizado ── */
        .contenido h1 {{
            color: #1a365d;
            font-size: 14pt;
            border-left: 4px solid #1a365d;
            padding-left: 10px;
            margin: 20px 0 12px 0;
            page-break-after: avoid;
        }}
        .contenido h2 {{
            color: #2d3748;
            font-size: 12pt;
            margin: 16px 0 10px 0;
            page-break-after: avoid;
        }}
        .contenido h3 {{
            color: #4a5568;
            font-size: 11pt;
            margin: 14px 0 8px 0;
            page-break-after: avoid;
        }}
        .contenido p {{
            margin: 8px 0;
        }}
        .contenido strong {{
            color: #1a365d;
        }}
        .contenido em {{
            color: #4a5568;
        }}

        /* ── Tablas ── */
        .contenido table {{
            width: 100%;
            border-collapse: collapse;
            margin: 14px 0;
            font-size: 9.5pt;
            page-break-inside: avoid;
        }}
        .contenido th, .contenido td {{
            border: 1px solid #cbd5e0;
            padding: 8px 10px;
            text-align: left;
        }}
        .contenido th {{
            background-color: #edf2f7;
            color: #1a365d;
            font-weight: 700;
        }}
        .contenido tr:nth-child(even) {{
            background-color: #f7fafc;
        }}

        /* ── Listas ── */
        .contenido ul, .contenido ol {{
            margin: 8px 0;
            padding-left: 24px;
        }}
        .contenido li {{
            margin: 4px 0;
        }}

        /* ── Bloques de código ── */
        .contenido code {{
            background-color: #edf2f7;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 9.5pt;
            color: #744210;
        }}
        .contenido pre {{
            background-color: #2d3748;
            color: #e2e8f0;
            padding: 12px;
            border-radius: 6px;
            overflow-x: auto;
            font-size: 9pt;
            line-height: 1.4;
            page-break-inside: avoid;
        }}
        .contenido pre code {{
            background: none;
            color: inherit;
            padding: 0;
        }}

        /* ── Bloques de cita ── */
        .contenido blockquote {{
            border-left: 4px solid #cbd5e0;
            margin: 12px 0;
            padding: 8px 16px;
            background-color: #f7fafc;
            color: #4a5568;
            font-style: italic;
        }}

        /* ── Separadores ── */
        .contenido hr {{
            border: none;
            border-top: 1px solid #e2e8f0;
            margin: 16px 0;
        }}

        /* ── Pie de página adicional ── */
        .footer-nota {{
            margin-top: 40px;
            padding-top: 12px;
            border-top: 1px solid #e2e8f0;
            font-size: 8pt;
            color: #a0aec0;
            text-align: center;
        }}
    </style>
</head>
<body>
    <div class="header-banner">
        <h1>MINUTA Y REPORTE CONSULTIVO | YANETH-IA</h1>
        <p class="subtitle">Consultoría en Gestión de Proyectos, PMO y Análisis Financiero de Inversiones</p>
        <p class="meta">Generado el: {fecha_hora} | Desarrollado por: Prof. Víctor Campos</p>
    </div>
"""

    # ── Procesar cada elemento del historial ─────────────────────
    for elemento in historial_filtrado:
        user_text = ""
        bot_text = ""

        if isinstance(elemento, dict):
            role = elemento.get("role", "")
            content = elemento.get("content", "")
            if role == "user":
                user_text = content
            elif role == "assistant":
                bot_text = content
        elif isinstance(elemento, (list, tuple)) and len(elemento) >= 2:
            user_text, bot_text = elemento[0], elemento[1]

        # Renderizar texto del usuario
        if user_text:
            md_html = markdown.markdown(
                user_text,
                extensions=['tables', 'fenced_code', 'nl2br']
            )
            html_content += f"""
            <div class="bloque-conversacion">
                <div class="rol-usuario">SOLICITUD DEL CLIENTE / USUARIO</div>
                <div class="contenido">{md_html}</div>
            </div>"""

        # Renderizar respuesta del asistente
        if bot_text:
            md_html = markdown.markdown(
                bot_text,
                extensions=['tables', 'fenced_code', 'nl2br']
            )
            html_content += f"""
            <div class="bloque-conversacion">
                <div class="rol-asistente">ENTREGABLE DE CONSULTORÍA (YANETH-IA)</div>
                <div class="contenido">{md_html}</div>
            </div>"""

    # Cerrar documento
    html_content += """
    <div class="footer-nota">
        Documento generado automáticamente por Yaneth-IA | Sujeto a revisión profesional.
    </div>
</body>
</html>"""

    # ── Generar PDF ─────────────────────────────────────────────
    try:
        pdf_path = os.path.abspath("Reporte_Consultoria_YanethIA.pdf")
        HTML(string=html_content).write_pdf(pdf_path)
        return pdf_path
    except Exception as e:
        print(f"Error al generar PDF: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
# INTERFAZ GRADIO — VERSIÓN OPTIMIZADA
# ═══════════════════════════════════════════════════════════════

with gr.Blocks(title="Yaneth-IA Executive Suite", css="""
    .gradio-container { max-width: 1100px !important; }
    .chatbot { min-height: 500px; }
""") as demo:

    gr.Markdown("""
    # 🤖 Yaneth-IA: Consultor en Gestión de Proyectos y Análisis Financiero
    > **Desarrollado por:** Prof. Víctor Campos | CI V-8270225
    >
    > *Especialista en PMO, CapEx/OpEx, EDT/WBS, Scrum, Kanban y análisis de rentabilidad de inversiones.*
    """)

    # ── Área de chat ──
    chatbot = gr.Chatbot(
        type="messages",
        label="💼 Mesa de Trabajo Consultiva",
        height=520,
        bubble_full_width=False,
        show_copy_button=True,
    )

    # ── Entrada de texto ──
    with gr.Row():
        msg = gr.Textbox(
            placeholder="Escribe tu consulta sobre PMO, CapEx, OpEx, EDT, riesgos, cronogramas...",
            label="📝 Entrada de Consulta",
            scale=5,
            lines=1,
            show_label=True,
        )
        btn_enviar = gr.Button("📤 Enviar", variant="primary", scale=1, size="lg")

    # ── Botones de control ──
    with gr.Row():
        btn_limpiar = gr.Button("🗑️ Limpiar Conversación", variant="stop", size="sm")
        btn_pdf = gr.Button("📄 Exportar Minuta a PDF", variant="secondary", size="sm")

    # ── Área de descarga ──
    gr.Markdown("---")
    with gr.Row():
        archivo_descarga = gr.File(
            label="📥 Descargar Reporte PDF Generado",
            interactive=False,
            visible=True,
        )

    # ── Eventos ──
    def limpiar_entrada():
        return ""

    def limpiar_chat():
        return []

    # Enviar con Enter o botón
    msg.submit(responder, [msg, chatbot], chatbot).then(limpiar_entrada, None, msg)
    btn_enviar.click(responder, [msg, chatbot], chatbot).then(limpiar_entrada, None, msg)

    # Limpiar chat
    btn_limpiar.click(limpiar_chat, None, chatbot, queue=False)

    # Exportar PDF
    btn_pdf.click(exportar_a_pdf, inputs=[chatbot], outputs=[archivo_descarga])


# ═══════════════════════════════════════════════════════════════
# LANZAMIENTO
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=10000,
        inline=False,
        show_error=True,
    )
