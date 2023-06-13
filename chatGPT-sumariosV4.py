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
chat = ChatOpenAI(temperature=0.3, model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY) # gpt-3.5-turbo "text-davinci-003" temperature=0
llm_predictor = LLMPredictor(llm=chat)


# Template de mensajes para el chat
CHAT_PROMPT_TEMPLATE_MESSAGES = [
    SystemMessage(content="Por favor, responde en español."),
    SystemMessage(content="Si respondes en inglés, serás penalizado."),
    SystemMessage(content="Eres un experto en el campo legal y jurídico, especialmente en Argentina."),
    SystemMessage(content="Estás aquí para realizar sumarios juridicos."),
    SystemMessage(content="Si un sumario no contiene 100 palabras como minimo, serás penalizado"),
    
    HumanMessagePromptTemplate.from_template(
        "Información sobre el contexto a continuación:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"

        "A continuación, se proporciona información sobre los sumarios jurídicos y los pasos necesarios para crear uno. Proporcione un resumen técnico jurídico que incluya la pena o absolución dictada por los jueces, los antecedentes y pruebas presentadas contra el acusado o acusados, las cuestiones legales en disputa y el análisis legal realizado por los jueces.\n\n"

        "2. Pasos para hacer sumarios:\n"
        "2.1 Primero se deben identificar cuántos sumarios existen el fallo:\n"
        "2.1.1 Por cada uno de los principales resuelve (pena o absolución) del tribunal, se debe hacer un sumario.\n"
        "2.2 Segundo, por cada sumario identificado en el punto anterior, se debe redactar un resumen técnico jurídico que contenga:\n"
        "2.2.1 La explicación detallada de la/s pena/s o absolución/es dictada/s por los jueces.\n"
        "2.2.2 Los antecedentes y pruebas presentadas contra el acusado/s.\n"
        "2.2.3 Las cuestiones legales en disputa\n"
        "2.2.4 El análisis legal realizado por los jueces.\n\n"
        
        "3 Características de los sumarios:\n"
        "3.1 El sumario debe interpretarse y comprenderse de manera independiente al resto de los sumarios.\n"
        "3.2 Un mismo hecho jurídico relevante no debe repetirse en otros sumarios.\n"
        "3.3 La identidad de los jueces no puede ser revelada y debe permanecer anónima.\n"
        "3.4 En el sumario cada acusado debe ser claramente identificado sin revelar su nombre y apellido\n"
        
        # "Dada la información sobre el contexto, responde lo siguiente:\n"
        # "(si no conoces la respuesta, responde con lo que sepas):\n"
        "{query_str}\n"

    ),
    HumanMessage(content=
                 "El formato de tu respuesta debe ser siempre, obligatoriamente, de esta forma: \n\n"
                 "Sumario: <sumario> \n"
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
        "Recuerda, cada sumario debe contener como minimo 100 palabras, si no es un error grave.\n"
        "Recuerda que una resolución judicial es importante cuándo hay una interpretación de las leyes o una interpretación del juez, o cuando se explica la pena o absolución del juez o jueces"
        "Debe haber un sumario para cada resolución judicial importante descartando el resto.\n"
        "El sumario debe explicar cual fue la pena o absolución.\n"
        "Si hay varios acusados en un sumario y reciben penas diferentes entonces debe haber un sumario separado para cada acusado.\n"
        "Si no es necesario hacer cambios, ignora estas instrucciones y repite la respuesta anterior.\n"
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
# query = "Da las clasificaciones de derecho a las que pertenece el texto: penal, penal economico, privado, procesal, procesal penal, publico o laboral? Dime el porque"
#query = 'Primero se requiere identificar cuantos sumarios jurídicos existen. Segundo, redactar cada sumario. Tercero, determinar las materias del derecho que se involucran en cada sumario.'
query = "Dado el fallo provisto en el contexto y el conocimiento para identificar cuántos sumarios existen en el fallo, redacta el o los sumarios"
# Realiza la búsqueda de similitud y obtiene los documentos más relevantes
respuesta_final = index.query(query, text_qa_template = CHAT_PROMPT, refine_template = CHAT_REFINE_PROMPT ) # Busca la pregunta a partir de los indices.

print("\n",respuesta_final)
