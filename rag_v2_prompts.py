from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate

# Prompt para la creación del grafo
system_prompt = (
    "# Knowledge Graph Instructions for GPT-4\n"
    "## 1. Overview\n"
    "You are a top-tier algorithm designed for extracting information in structured "
    "formats to build a knowledge graph.\n"
    "Capture every single piece of information from the text without " # 1
    "sacrifing accuracy. Do not add any information that is not explicitly "
    "mentioned in the text\n"
    "- **Nodes** represent entities and concepts.\n"
    "- The aim is to achieve simplicity and clarity in the knowledge graph, making it\n"
    "accessible for a vast audience.\n"
    "## 2. Labeling Nodes\n"
    "- **Consistency**: Ensure you use available types for node labels.\n"
    "Ensure you use basic or elementary types for node labels.\n"
    "- For example, when you identify an entity representing a person, "
    "always label it as **'person'**. Avoid using more specific terms "
    "like 'mathematician' or 'scientist'"
    "  - **Node IDs**: Never utilize integers as node IDs. Node IDs should be "
    "names or human-readable identifiers found in the text. "
    "Node IDs should be exactly as they appear in the text. This means that accents, " # 2
    "diacritical marks, and capitalization must be retained in their original form. "
    "The goal is to ensure that the extracted entities are a perfect match to their appearance in the original text. \n"
    "- **Relationships** represent connections between entities or concepts.\n"
    "Ensure consistency and generality in relationship types when constructing "
    "knowledge graphs. The relationships should be as long as necessary to capture the most amount of information possible. " # 3
    "Relationships should be in the same language as they appear in the text."
    "## 3. Coreference Resolution\n"
    "- **Maintain Entity Consistency**: When extracting entities, it's vital to "
    "ensure consistency.\n"
    'If an entity, such as "John Doe", is mentioned multiple times in the text '
    'but is referred to by different names or pronouns (e.g., "Joe", "he"),'
    "always use the most complete identifier for that entity throughout the "
    'knowledge graph. In this example, use "John Doe" as the entity ID.\n'
    "Remember, the knowledge graph should be coherent and easily understandable, "
    "so maintaining consistency in entity references is crucial.\n"
    "## 4. Strict Compliance\n"
    "Adhere to the rules strictly. Non-compliance will result in termination."
)

GRAPH_GENERATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            system_prompt,
        ),
        (
            "human",
            (
                "Tip: Make sure to answer in the correct format, and do "
                "not include any explanations. "
                "Use the given format to extract information from the "
                "following input: {input}"
            ),
        ),
    ]
)

# Prompt para la creación de la consulta Cypher v1
CYPHER_GENERATION_TEMPLATE_v1 = """Task: Generate a Cypher statement to query a neo4j graph database.
Instructions:
Use only the provided node properties, relationship properties and relationships from the following schema.
Use only MATCH and RETURN statements.
The MATCH statement should have the following format, where *entity here* should be replaced with the main entity extracted from the question:
MATCH (d:Document)-[:MENTIONS]->(c:Concept *curly bracket*id: *entity here**curly bracket*)
The entity extracted from the question should never be capitalized and must be written with the correct accent marks.
For example, if the entity in the question is "Coche Electrico", use "coche eléctrico" in the statement.
The RETURN statement should contain every property of the document besides the id.

Schema:
{schema}

Notes:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Question:
{question}"""


# Prompt para la creación de la consulta Cypher v2
CYPHER_GENERATION_TEMPLATE_v2 = """Task: Generate a Cypher statement to query a neo4j graph database.
Instructions for the statement:
Use only the provided node properties, relationship properties and relationships from the following schema:

Schema:
{schema}

Ignore "Document" nodes, do not use them in any case.
The nodes of the RETURN statement should have a meaningful name.
The entity extracted from the question should never be capitalized and must be written with the correct accent marks.
For example, if the entity in the question is "Coche Electrico", use "coche eléctrico" in the statement.

Notes:
Do not include any explanations or apologies in your responses.
Do not respond to any questions that might ask anything else than for you to construct a Cypher statement.
Do not include any text except the generated Cypher statement.

Question:
{question}"""

CYPHER_GENERATION_PROMPT_V2 = PromptTemplate(
    input_variables=["schema", "question"], template=CYPHER_GENERATION_TEMPLATE_v2
)

# Prompt para la creación de la respuesta
CYPHER_QA_TEMPLATE = """You are an assistant that helps to form nice and human understandable answers.
The information part contains the provided information that you must use to construct an answer.
The provided information is authoritative, you must never doubt it or try to use your internal knowledge to correct it.
Make the answer sound as a response to the question.
The answer must have every information provided in the context.
Respond in a natural and conversational manner without using quotation marks to highlight the key concepts or phrases provided in the context.
For example, instead of saying 'La clave del éxito es "trabajo duro" y "dedicación".' say 'La clave del éxito es el trabajo duro y la dedicación.'
Do not mention that you based the result on the given information.
Here is an example:

Question: Which managers own Neo4j stocks?
Context:[manager:CTL LLC, manager:JANE STREET GROUP LLC]
Helpful Answer: CTL LLC, JANE STREET GROUP LLC owns Neo4j stocks.

Follow this example when generating answers.
If the provided information is empty, say that you don't know the answer.
Information:
{context}

Question: {question}
Helpful Answer:"""

CYPHER_QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"], template=CYPHER_QA_TEMPLATE
)