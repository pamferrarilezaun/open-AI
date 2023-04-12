# Documentacion de la libreria langchain: https://langchain.readthedocs.io/en/latest/getting_started/getting_started.html
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain import PromptTemplate

# Pinecone es una plataforma de indexacion y recuperacion de vectores diseniada para manejar 
# grandes cantidades de datos de alta dimensionalidad. 
# Permite indexar vectores de datos y recuperar los vectores mas cercanos a una consulta dada en tiempo real, 
# lo que la hace util en el procesamiento del lenguaje natural.
import pinecone

# Busca los txt de la carpeta a la que corresponde
ruta = 'data/prueba.txt'
loader = UnstructuredFileLoader(ruta)

# Lee los txt
data = loader.load()
# print(data[0])

# Te cuenta con cuantos documentos estas trabajando y ademas te cuenta la cantidad de caracteres que tiene c/ doc.
print (f'Hay {len(data)} documento(s) en data')
print (f'Hay {len(data[0].page_content)} caracteres en el documento')

# Divide los datos (en este caso el txt) en fragmentos mas pequeños
# Se divide en fragmentos con un tamanio de 1000 y el segundo parametro es para que no superpongan los fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Se corta cada texto para que queden por separado
texts = text_splitter.split_documents(data)
print (f'Ahora hay {len(texts)} documentos') # Te cuenta cuantos documentos diferentes se formaron.
# print("El primer texto es:", texts[0]) # Imprime el primer documento.

# Se crean embeddings para cada uno de los documentos para la busqueda semantica
OPENAI_API_KEY = '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'
PINECONE_API_KEY = 'a72ad1a6-c685-459d-8adf-56cc1dc07c91'
PINECONE_API_ENV = 'us-east-1-aws'
# Si el server esta en estados unidos creo que deberia ir la siguiente linea: PINECONE_API_ENV = 'us-east1-gcp'

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# iinicializacion de pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)

# Se crea el nombre del indice 
index_name = "langchain2"
# Se elimina en caso de ser necesario
pinecone.delete_index(index_name)
# Se crea el nuevo indice con la dimension de 1536 porque es el vector que acepta el modelo de openAI
pinecone.create_index(index_name, dimension=1536)


# Utiliza el metodo Pinecone.from_texts() para crear embeddings de los textos en la lista de textos,
# por cada objeto que tiene un atributo page_content realiza lo anterior. Estos embeddings creados 
# se almacenan en un indice de Pinecone con el nombre especificado por index_name.
docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)
# print(docsearch)

# print(docs)

# A partir de aca se agrega langchain
# Primer parametro: En este caso se utiliza GPT
# Segundo parametro: una temperatura de 0; Los valores mas altos como 0,8 haran que la salida sea mas aleatoria
# mientras que los valores mas bajos como 0,2 la haran mas enfocada y determinista.
#  Tercer parametro: son las credenciales de la API
llm = OpenAI(model_name = "gpt-3.5-turbo", temperature=0.5, openai_api_key=OPENAI_API_KEY)

# Definimos el prompt personalizado
prompt_template = """Context: {context}
Query: {question}"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Con este metodo se puede realizar dos tareas: utilizar el prompt y llamar al modelo de generacion de texto (openAi en este caso)
chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

# La pregunta que va a buscar la IA en los textos.
query = "¿A cuál clasificacion pertenece: derecho penal o derecho penal economico o derecho privado o derecho procesal o derecho procesal penal o derecho publico o derecho laboral? Explica el porque. Si no puedes determinarlo con exactitud, da dos clasificaciones posibles."


# Busca todos fragmentos dentro de los documentos mas similares a la pregunta que se hizo.
docs = docsearch.similarity_search(query)

context = '\n'.join([doc.page_content for doc in docs])
# print(context)

# Aca se ingresan los documentos relevantes encontrados en la sentencia anterior y ademas la query
respuesta_final = chain.run(input_documents=docs, question=query, context = context)
print("respuesta final", respuesta_final)








