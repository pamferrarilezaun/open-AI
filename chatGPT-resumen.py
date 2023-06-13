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
chat = ChatOpenAI(temperature=0.2, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY)
llm_predictor = LLMPredictor(llm=chat)


# Template de mensajes para el chat
CHAT_PROMPT_TEMPLATE_MESSAGES = [
    SystemMessage(content="Por favor, responde en español."),
    SystemMessage(content="Si respondes en inglés, serás penalizado."),
    SystemMessage(content="Eres un experto en el campo legal y jurídico, especialmente en Argentina."),
    SystemMessage(content="Estás aquí para realizar resumenes técnicos juridicos."),
    
    HumanMessagePromptTemplate.from_template(
        "Información sobre el contexto a continuación:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"

        "El juez o tribunal a partir de los hechos explicados en el fallo toma decisiones sobre la aplicación de normas o leyes relativas a esos hechos."
        "En base a esos hechos, las normas o leyes fueron aplicadas y la interpretación de esas normas el juez o tribunal dicta sentencia."
        
        "{query_str}\n"

    ),
    HumanMessage(content=
                 "El formato de tu respuesta debe ser siempre, obligatoriamente, de esta forma: \n\n"
                 "Interpretación del texto: <Interpretación del texto> \n"
                 #"Materias del derecho: <Materias>\n"
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
        "Dado el nuevo contexto y usando lo mejor de tu conocimiento, mejora la respuesta anterior.\n"
        "Recuerda, si respondes en inglés será un error grave.\n"
        "Si no conoces la respuesta escribe lo que sepas del fallo."
        "Si no tiene información suficiente para responder no lo escribas en la respuesta."
        "Recuerda que debes responder como un abogado experto en fallos judiciales\n"
        "Es grave que la salida no contenga como minimo 120 palabras.\n"
        "Si no es necesario hacer cambios, ignora estas instrucciones y repite la respuesta anterior."
    ),
]

CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

# Definimos el prompt
# Esta es la máxima cantidad que permite de ingreso de un texto
max_input_size = 4096  #4096
# Esta es la máxima cantidad de tokens que va a tener la respuesta
# En algun caso ser preferible que la respuesta sea concisa (num_output = 256) y que la pregunta tenga la mayor
# cantidad posible de contexto. Como en este caso es solo clasificación esta lógica va a servir, con los sumarios
# se tiene que aplicar otra estrategia.
num_output = 1800
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
query ="Explica las penas o absoluciones dictadas por el juez o tribunal para cada acusado. Estan al final del fallo. Si existen sobreseimientos también escríbelos." 
respuesta1 = index.query(query, text_qa_template = CHAT_PROMPT, refine_template = CHAT_REFINE_PROMPT ) # Busca la pregunta a partir de los indices.
print("\n",respuesta1)

query = "Explica detalladamente qué hechos delictivos fueron cometidos por cada acusado. Si no tienes información habla de los hechos jurídicos generales del fallo"
respuesta2 = index.query(query, text_qa_template = CHAT_PROMPT, refine_template = CHAT_REFINE_PROMPT ) # Busca la pregunta a partir de los indices.
print("\n",respuesta2)

query = "Nombra cuáles son las normas o leyes aplicadas o citadas en el fallo. Explica porque se mencionan esas normas o leyes."
respuesta3 = index.query(query, text_qa_template = CHAT_PROMPT, refine_template = CHAT_REFINE_PROMPT ) # Busca la pregunta a partir de los indices.
print("\n",respuesta3)

# query = "Hace un resumen extensivo sobre los temas importantes del fallo"
# respuesta4 = index.query(query, text_qa_template = CHAT_PROMPT, refine_template = CHAT_REFINE_PROMPT ) # Busca la pregunta a partir de los indices.
# print("\n",respuesta4)

