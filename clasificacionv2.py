from langchain.llms import OpenAI
from llama_index import LLMPredictor
from langchain.chat_models import ChatOpenAI
from llama_index import GPTSimpleVectorIndex, PromptHelper, ServiceContext

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
chat = OpenAI(temperature=0.4, model_name="text-davinci-003") # gpt-3.5-turbo "text-davinci-003" temperature=0
llm_predictor = LLMPredictor(llm=chat)

# Definimos el prompt
# Esta es la máxima cantidad que permite de ingreso de un texto
max_input_size =4096  #4096
# Esta es la máxima cantidad de tokens que va a tener la respuesta
# En este caso va a ser de 256 porque es preferible que la respuesta sea concisa y que la pregunta tenga la mayor
# cantidad posible de contexto. Como en este caso es solo clasificación esta lógica va a servir, con los sumarios
# se tiene que aplicar otra estrategia.
num_output = 1600
# un poco overlap es bueno porque de esta forma se logra que no queden desacopladas las partes y se puedan relacionar
max_chunk_overlap = 20
# crea una instancia de PromptHelper con las caracteristicas anteriores, esto se envia al llm
prompt_helper = PromptHelper(max_input_size, num_output, max_chunk_overlap)

# Se le pasa el llm y el prompt, define que el llm va a ser gpt-3.5- turbo y las restricciones del prompt
service_context = ServiceContext.from_defaults(llm_predictor=llm_predictor, prompt_helper=prompt_helper)

# En este caso se va a crear un indice pero con los valores customizados porque contiene el service_context
# o sea con el prompt definido y el llm
index = GPTSimpleVectorIndex.from_documents(
    documents, service_context=service_context
)

# La pregunta que va a buscar la IA en los textos.
query = "Primero identificar cuantos sumarios tiene el texto: Se debe hacer un sumario por cada resuelve importante del juez. Segundo redactar cada sumario explicando las cuestiones legales en disputa, el analisis legal del juez o jueces  y la pena para el acusado"
# query = "Realizar un sumario juridico por cada hecho relevante del fallo. Se debe esplicar como un informe tecnico legal los antecedentes del caso, las cuestiones legales en disputa, el análisis legal, y principalmente la decisión del juez o jueces y la pena para el acusado. "

# Realiza la búsqueda de similitud y obtiene los documentos más relevantes
respuesta_final = index.query(query) # Busca la pregunta a partir de los indices

print("respuesta final", respuesta_final)