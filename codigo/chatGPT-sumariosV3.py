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
        "1. Un sumario es un resumen tecnico juridico sobre la decisión emitida por un juez o jueces después de considerar y evaluar todos los hechos y pruebas presentadas en un caso legal."
        "2. Un sumario explica las resoluciones finales que dictaminan quién tiene la razón en el caso y qué acciones o consecuencias deben seguirse."
        
        "3. Un hecho jurídico relevante es aquel que es importante para determinar si se ha cometido una infracción de la ley. Es decir, son los hechos que están estrechamente vinculados con los temas legales que están en disputa."
        "4. La doctrina se refiere lo que resolvió un juez o jueces sobre un hecho juridico relevante. En otras palabras, la doctrina es la/s pena/s para el acusado o la/s absolucion/es sobre los hechos juridicos relevantes."
        "5. Los antecedentes es la explicación de los hechos jurídicos relevantes y las pruebas presentadas contra el acusado"
        "6. Las cuestiones legales en disputa son los temas legales que fueron objeto de controversia en el caso"
        "7. El análisis legal son los argumentos legales presentados por ambas partes y la interpretación que hizo el juez o jueces"
        "8. Las partes involucradas son quienes estraron en conflicto."
        
        "9. Pasos para hacer sumarios:"
        "9.1 Primero se deben identificar cuantos sumarios tiene el texto:"
        "9.1.1 Por cada una de los principales resuelve del juez o tribunal se debe hacer un sumario. Solo para los resuelve de importancia"
        "9.1.2 Se deben hacer sumarios para los hechos juridicos relevantes"
        "9.2 Segundo, por cada sumario identifacado en el punto anterior se debe redactar un resumen técnico jurídico que detalle:"
        "9.2.1 La pena o absolución por parte del juez o jueces y quienes fueron las partes involucradas"
        "9.2.2 Los antecedentes y las pruebas presentadas contra el acusado"
        "9.2.3 Las cuestiones legales en disputa y el analisis legal hecho por los jueces." #, los antecedentes, las cuestiones legales, el análisis legal y las concecuencias.
        "9.3 El texto legal que surge como sumario tiene que interpretarse y comprenderse de manera independiente al resto de los sumarios"
        
        # "Información sobre las materias del derecho existentes:"
        # "Las materias del derecho son las siguientes: Derecho Privado y Comunitario\n2.Derecho de Daños\n3.Derecho Procesal\n4.Derecho Comparado\n5.Derecho Laboral\n6.Derecho Penal\n7.Derecho Público\n8.Derecho Procesal Penal\n9.Derecho Laboral Actualidad\n10.Derecho Penal Económico"
        
        # "Dada la información sobre el contexto, responde lo siguiente:\n"
        # "(si no conoces la respuesta, responde con lo que sepas):\n"
        "{query_str}\n"

        "Por favor, las materias del derecho de cada sumario individual deben estar listadas aparte"
        "Por favor, no incluyas nombres propios"

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
#query = 'Primero se requiere identificar cuantos sumarios jurídicos existen. Segundo, redactar cada sumario. Tercero, determinar las materias del derecho que se involucran en cada sumario.'
query = "Dado el conocimiento sobre como hacer un sumario, y dado el fallo provisto en el contexto: Redacta en forma extensiva el o los sumarios correspondientes."
# Realiza la búsqueda de similitud y obtiene los documentos más relevantes
respuesta_final = index.query(query, text_qa_template = CHAT_PROMPT, refine_template = CHAT_REFINE_PROMPT ) # Busca la pregunta a partir de los indices.

print("\n",respuesta_final)
#Realiza una lista de los principales hechos jurídicos del texto. 