from langchain.prompts import ChatPromptTemplate

# Troceado semántico LLMs
LLM_SPLITTER_PROMPT = "Trocea el siguiente texto siguiendo la técnica de troceado semántico. Añade >>> y <<< alrededor de cada trozo: '{text}'"

# Etiquetado
SYSTEM_LABEL_PROMPT = "Responde SÓLO con el nombre de la etiqueta, sin añadir la descripción"

USER_LABEL_PROMPT = (
    "Eres un experto en entendiendo la normativa de la Universidad de La Laguna.\n"
    "Tu trabajo es etiquetar correctamente el siguiente extracto de la normativa con una de las siguientes etiquetas:\n"
        "Introducción: Una sección introductoria que proporciona una visión general del propósito y alcance de las bases de ejecución presupuestaria.\n",
        "Marco Legal: Una descripción de las leyes, reglamentos y normativas que rigen la gestión presupuestaria de la entidad.\n",
        "Objetivos: Una declaración de los objetivos y metas que se buscan alcanzar mediante la gestión y ejecución del presupuesto.\n",
        "Procedimientos de Elaboración del Presupuesto: Detalles sobre el proceso de elaboración del presupuesto, incluyendo la participación de diferentes áreas o departamentos, los plazos involucrados y los criterios utilizados para asignar recursos.\n",
        "Normas de Ejecución: Reglas y procedimientos específicos para la ejecución del presupuesto, como la autorización de gastos, la contratación pública, el control de pagos, entre otros.\n",
        "Control y Seguimiento: Procedimientos para el control y seguimiento del presupuesto, incluyendo la elaboración de informes periódicos, auditorías internas o externas, y mecanismos de rendición de cuentas.\n",
        "Modificaciones Presupuestarias: Procedimientos para realizar modificaciones al presupuesto inicial, como transferencias de créditos o incorporaciones de remanentes.\n",
        "Disposiciones Adicionales: Otras disposiciones relevantes para la gestión y ejecución del presupuesto, como la gestión de deuda, el manejo de contingencias, entre otros.\n"
    "\nExtracto:\n"
)

# Respuesta
QA_PROMPT_TEMPLATE = """Responde a la pregunta basándote sólo en el siguiente contexto:

{context}

---

Responde a la pregunta basándote en el contexto de arriba: {question}
"""

QA_PROMPT = ChatPromptTemplate.from_template(QA_PROMPT_TEMPLATE)