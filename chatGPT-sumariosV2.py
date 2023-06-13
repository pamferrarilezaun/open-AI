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
chat = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo") # gpt-3.5-turbo "text-davinci-003" temperature=0
llm_predictor = LLMPredictor(llm=chat)


# Template de mensajes para el chat
CHAT_PROMPT_TEMPLATE_MESSAGES = [
    SystemMessage(content="Por favor, responde en español."),
    SystemMessage(content="Si respondes en inglés, serás penalizado."),
    SystemMessage(content="Eres un experto en el campo legal y jurídico, especialmente en Argentina."),
    SystemMessage(content="Estás aquí para realizar sumarios juridicos."),
    HumanMessagePromptTemplate.from_template(
        "Información sobre el contexto a continuación:\n"
        "---------------------\n"
        "{context_str}\n"
        "---------------------\n"

        "Información sobre sumarios jurisprudenciales:"
        "1. En un fallo judicial existen uno o varios sumarios. Un sumario es resumen tecnico juridico."
        "2- Para poder identificar la cantidad de sumarios en un fallo, se debe:"
        "2.1 Identificar las doctrinas de un fallo judicial: La doctrina se refiere a la resolución que se le dio a una controversia legal en un fallo judicial. Es decir, cómo el tribunal resuelve una determinada situación legal basada en un hecho jurídico o en la interpretación de una norma."
        "2.2 Identificar los hechos jurídicos relevantes. Un hecho juridico relevante es el que es necesario para determinar si se ha producido una infracción de la ley. Los hechos jurídicos relevantes son los que están directamente relacionados con los asuntos legales en cuestión"
        "3- Debe existir una única doctrina por cada sumario. Es un error grave incluir más de una doctrina en un sumario."
        "4- Es un error grave no hacer un sumario con los hechos jurídicos relevantes."
        "5 El ultimo sumario debe contener la sentencia explicada en detalle:"
        "5.1 Se debe escribir si la persona fue absuelta o fue condenada"
        "5.2 Si fue condenado, se debe escribir la condena que recibió ya sea civil o penal"
        "6- La redacción de un sumario debe poseer un formato adecuado:"
        "6.1 Se debe redactar un texto con lenguaje tecnico juridico donde se explique la decision final del juez o de los jueces acerca del demandado o imputado: si hubo condena o absolución y el porque."
        "6.2 Se debe establecer como llegaron el o los jueces a la condena o absolución y las posibles concecuencias para el demandado o imputado."
        "6.3 Cada sumario debe contener una descripción exhaustiva de los hechos jurídicos."
        "6.4 Cada sumario debe contener las pruebas presentadas por las partes."
        "6.4 Se deben describir cuáles fueron las cuestiones legales en disputa, o sea que cuestión produjo controversia."
        "6.5 También se deben mencionar los argumentos legales que se presentaron en favor del demandado o imputado y como fue la interpretacion del él o los jueces."
        
        # "Informacion adicional:"
        # "Un hecho juridico es cualquier acontecimiento apto para tener relevancia jurídica. Se prevé la hipótesis de la verificación del hecho y la posibilidad de que este hecho una vez que se produzca, adquiera relevancia jurídica. El hecho, al verificarse, lleva a efecto aquello previsto por la ley."
        
        "Información sobre las materias del derecho existentes:"
        "Las materias del derecho son las siguientes: Derecho Privado y Comunitario\n2.Derecho de Daños\n3.Derecho Procesal\n4.Derecho Comparado\n5.Derecho Laboral\n6.Derecho Penal\n7.Derecho Público\n8.Derecho Procesal Penal\n9.Derecho Laboral Actualidad\n10.Derecho Penal Económico"
        
        # "Dada la información sobre el contexto, responde lo siguiente:\n"
        # "(si no conoces la respuesta, responde con lo que sepas):\n"
        "{query_str}\n"

        "Por favor, las materias del derecho de cada sumario individual deben estar listadas aparte"
        "Por favor, no incluyas nombres propios"

    ),
    # HumanMessage(content=
    #              "El formato de tu respuesta debe ser siempre, obligatoriamente, de esta forma: \n\n"
    #              "Sumario: <sumario> \n"
    #              "Materias del derecho: <Materias>\n"
    #             )
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
        "Recuerda, si respondes en inglés será un error grave"
        "La salida debe respetar el formato que fue especificado"
        "Explayate tanto como sea posible en cada sumario, y enumeralos"
        "Si no es necesario hacer cambios, ignora estas instrucciones y repite la respuesta anterior."
        # "La salida no debe ser una lista. La salida debe ser un texto legal completo"
        # "Tener en cuenta el formato de la salida"
    ),
]

CHAT_REFINE_PROMPT_LC = ChatPromptTemplate.from_messages(CHAT_REFINE_PROMPT_TMPL_MSGS)
CHAT_REFINE_PROMPT = RefinePrompt.from_langchain_prompt(CHAT_REFINE_PROMPT_LC)

# Definimos el prompt
# Esta es la máxima cantidad que permite de ingreso de un texto
max_input_size = 4096  #4096

# Esta es la máxima cantidad de tokens que va a tener la respuesta
# En el caso de 256 es porque la respuesta debe ser concisa y la pregunta debe tener la mayor
# cantidad posible de contexto, como es el caso de clasificación. Tiene otra lógica para realizar sumarios y
# no va a servir lo anterior, en los sumarios la respuesta debe ser larga con lo cuál se debe aumento el valor a todo lo necesario.
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
#query = 'Primero se requiere identificar cuantos sumarios jurídicos existen. Segundo, redactar cada sumario. Tercero, determinar las materias del derecho que se involucran en cada sumario.'
query = "Dado el conocimiento sobre lo que es un sumario, y dado el fallo provisto en el contexto: Identifica todos los sumarios y redactalos. Luego haz un sumario sobre la sentencia del juez. Por cada sumario nombra la o las materias del derecho con la que se relaciona."
# Realiza la búsqueda de similitud y obtiene los documentos más relevantes
respuesta_final = index.query(query, text_qa_template = CHAT_PROMPT, refine_template = CHAT_REFINE_PROMPT ) # Busca la pregunta a partir de los indices.

print("\n",respuesta_final)
#Realiza una lista de los principales hechos jurídicos del texto. 