from langchain.llms import OpenAI
from llama_index import LLMPredictor
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, PromptHelper, ServiceContext

from langchain.prompts.chat import (
    AIMessagePromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    SystemMessage,
    AIMessage,
    HumanMessage
)

from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt

#GPTSimpleVectorIndex es una clase que construye un índice de búsqueda basado en vectores
#  utilizando embeddings generados por el modelo GPT. 
# SimpleDirectoryReader es una clase que se utiliza para leer documentos de un directorio
from llama_index import GPTSimpleVectorIndex, SimpleDirectoryReader

# Busca los txt de la carpeta a la que corresponde
ruta = 'data'

# Esto usa LlamaIndex
documents = SimpleDirectoryReader(ruta).load_data()
# print(documents)

# Clave de openAI
OPENAI_API_KEY = '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'

# Vamos a customizar para nuestro caso
# define LLM, en este caso vamos a utilizar "gpt-3.5-turbo"
chat = ChatOpenAI(temperature=0.6, model_name="gpt-3.5-turbo") # gpt-3.5-turbo "text-davinci-003" temperature=0
llm_predictor = LLMPredictor(llm=chat)


# Template de mensajes para el chat
# Template de mensajes para el chat
CHAT_PROMPT_TEMPLATE_MESSAGES = [
    SystemMessage(content="Por favor, responde en español."),
    SystemMessage(content="Si respondes en inglés, serás penalizado."),
    SystemMessage(content="Eres un experto en el campo legal y jurídico, especialmente en Argentina."),
    SystemMessage(content="Estás aquí para responder preguntas legales."),
    SystemMessage(content="Debes redactar un texto jurídico correcto, no un formato de lista, si no es un error grave"),
    HumanMessagePromptTemplate.from_template(
        "Información sobre el contexto a continuación:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"
        "Conociendo lo siguiente:\n"
        "1- Eres un Juez de la Suprema Corte de Justicia de la Nación Argentina.\n"
        "2- Eres un experto en realizar resúmenes técnicos/sumarios jurídicos de fallos judiciales.\n"
        "3- En un sumario se conocen todas las acciones del fallo ordenadas judicialmente.\n"
        "4- Características principales de un sumario:\n"
        "   - Objetividad: El sumario debe presentar la información de manera clara y objetiva, evitando interpretaciones o juicios de valor.\n"
        "   - Coherencia: El sumario debe seguir un orden, información de manera estructurada.\n"
        "   - Identificación de las ideas principales: El sumario debe destacar las ideas principales del texto original, identificando las afirmaciones más relevantes y los argumentos más importantes.\n"
        "   - Omisión de detalles irrelevantes: El sumario debe omitir detalles que no son esenciales para la comprensión del texto.\n"
        "5- Partes de un sumario:\n"
        "   - Identificación del caso: se incluyen los nombres de las partes involucradas y/o el número del expediente y/o el tribunal que dictó la sentencia.\n"
        "   - Antecedentes: se presenta un resumen de los hechos relevantes del caso y las pruebas presentadas por las partes.\n"
        "   - Cuestiones legales en disputa: se identifican las cuestiones legales que fueron objeto de controversia en el caso.\n"
        "   - Análisis legal: se explican los argumentos legales presentados por las partes y la interpretación que hizo el juez o tribunal de las leyes aplicables al caso.\n"
        "   - Decisión: se presenta la decisión final del juez o tribunal y se explica cómo llegaron a ella.\n"
        "   - Consecuencias: se explican las posibles consecuencias del fallo y cómo puede afectar a las partes involucradas y a la jurisprudencia en general.\n"
        "Dadas las siguientes materias del derecho:"
        "1.Derecho Privado y Comunitario\n2.Derecho de Daños\n3.Derecho Procesal\n4.Derecho Comparado\n5.Derecho Laboral\n6.Derecho Penal\n7.Derecho Público\n8.Derecho Procesal Penal\n9.Derecho Laboral Actualidad\n10.Derecho Penal Económico"
        "Dada la información sobre el contexto, responde lo siguiente:\n"
        " Es obligatorio que el contenido esté redactado como oraciones y párrafos, debe ser un texto juríridco técnico"
        "(si no conoces la respuesta, responde con lo que sepas):\n"
        "{query_str}\n"
    ),
    HumanMessage(content=
                 "El formato de tu respuesta debe ser siempre, obligatoriamente, de esta forma: \n\n"
                 "Sumario: <sumario> \n"
                 "Materias del derecho: <Materias>\n"
                )
]

# template con objeto de langchain
CHAT_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_PROMPT_TEMPLATE_MESSAGES)
#template con objeto de llama index
CHAT_PROMPT = QuestionAnswerPrompt.from_langchain_prompt(CHAT_PROMPT_LC)

CHAT_REFINE_PROMPT_TMPL_MSGS = [
    HumanMessagePromptTemplate.from_template("{query_str}"),
    AIMessagePromptTemplate.from_template("{existing_answer}"),
    HumanMessagePromptTemplate.from_template(
        "Tenemos la oportunidad de mejorar la respuesta anterior "
        "(en caso de ser necesario) con mayor contexto.\n"
        "------------\n"
        "{context_msg}\n"
        "------------\n"
        "Dado el nuevo contexto y usando lo mejor de tu conocimiento, mejora la respuesta anterior. "
    "Si no puedes mejorarla, solo repitela."
    "Recuerda, si respondes en inglés será un error grave"
    "Debes hacer una lista"
    
    ),
]

CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

# Definimos el prompt
# Esta es la máxima cantidad que permite de ingreso de un texto
max_input_size = 4096  #4096
# Esta es la máxima cantidad de tokens que va a tener la respuesta
# En este caso va a ser de 256 porque es preferible que la respuesta sea concisa y que la pregunta tenga la mayor
# cantidad posible de contexto. Como en este caso es solo clasificación esta lógica va a servir, con los sumarios
# se tiene que aplicar otra estrategia.
num_output = 1500
# un poco overlap es bueno porque de esta forma se logra que no queden desacopladas las partes y se puedan relacionar
max_chunk_overlap = 20
# crea una instancia de PromptHelper con las caracteristicas anteriores, esto se envia al llm
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Se le pasa el llm y el prompt, define que el llm va a ser gpt-3.5- turbo y las restricciones del prompt
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# En este caso se va a crear un indice pero con los valores customizados porque contiene el service_context
# o sea con el prompt definido y el llm
index = GPTSimpleVectorIndex.from_documents(
    documents, service_context=service_context, 
)

# La pregunta que va a buscar la IA en los textos.
# query = "Da las clasificaciones de derecho a las que pertenece el texto: penal, penal economico, privado, procesal, procesal penal, publico o laboral? Dime el porque"
query = 'Realizar un resumen técnico jurídico que contenga: identificación del caso, antecedentes, Cuestiones legales en disputa, Análisis legal, Decisión del juez y Consecuencias. Por ultimo nombra todas las materias del derecho que intervienen.'
# Realiza la búsqueda de similitud y obtiene los documentos más relevantes
respuesta_final = index.query(query, text_qa_template = CHAT_PROMPT, refine_template = CHAT_REFINE_PROMPT) # Busca la pregunta a partir de los indices

print("\n",respuesta_final)
#Realiza una lista de los principales hechos jurídicos del texto. 