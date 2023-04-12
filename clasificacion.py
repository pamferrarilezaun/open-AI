from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains.question_answering import load_qa_chain


# Busca los txt de la carpeta a la que corresponde
ruta = 'data/prueba.txt'
loader = TextLoader(ruta, encoding='utf-8')

# Lee los txt
data = loader.load()

print("El tipo de dato es", type(data))

# Te cuenta con cuantos documentos estas trabajando y ademas te cuenta la cantidad de caracteres que tiene c/ doc.
print (f'Hay {len(data)} documento(s) en data')
print (f'Hay {len(data[0].page_content)} caracteres en el documento')

# Divide los datos (en este caso el txt) en fragmentos mas pequeños
# Se divide en fragmentos con un tamanio de 1000 y el segundo parametro es para que no superpongan los fragmentos
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# Se corta cada texto para que queden por separado
docs = text_splitter.split_documents(data)

# Clave de openAI
OPENAI_API_KEY = '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'
# Crea una instancia de un objeto de embeddings utilizando la API de OpenAI.
# Los embeddings son representaciones vectoriales de palabras o frases. Estos vectores tienen 
# la propiedad de capturar cierta información semántica y sintáctica de las palabras, 
# lo que permite realizar operaciones matemáticas para obtener información útil.
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Crea un índice de búsqueda de vectores de embeddings de documentos.
# La variable docs es una lista de documentos o textos que se han preprocesado
# La variable embeddings es una matriz donde cada fila representa el 
# vector de embeddings correspondiente a un documento en la lista docs.
# La función from_documents toma estos documentos y embeddings 
# y crea un índice invertido que se puede utilizar para buscar documentos similares. 
# El índice se construye utilizando la técnica de Locality Sensitive Hashing o LSH) y otras técnicas
# Una vez que se ha creado el índice, se puede utilizar para encontrar documentos similares a una consulta específica\
# en función de la similitud de sus embeddings.
db = FAISS.from_documents(docs, embeddings)

# A partir de aca se agrega langchain
# Primer parametro: En este caso se utiliza como modelo GPT 3.5 turbo
# Segundo parametro: una temperatura de 0; Los valores mas altos como 0,8 haran que la salida sea mas aleatoria
# mientras que los valores mas bajos como 0,2 la haran mas enfocada y determinista.
#  Tercer parametro: son las credenciales de la API
llm = OpenAI(model_name = "gpt-3.5-turbo", temperature=0.5, openai_api_key=OPENAI_API_KEY)

# Definimos el prompt personalizado. Un prompt es la información semántica que uno le envía al modelo.
# En este caso va a ser el fallo, la pregunta, y las etiquetas a clasificar.
prompt_template = """

Context: {context}

Teniendo en cuenta las siguientes clasificaciones posibles:
1-penal
2-penal economico
3-privado
4-procesal
5-procesal penal
6-publico
9-laboral

Query: {question}

Tu respuesta DEBE tener el siguiente formato:
-Posible clasificacion: [posibles categorias enumeradas]
-Explicacion: [motivo]

"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Con este metodo se puede realizar dos tareas: utilizar el prompt y llamar al modelo de generacion de texto (openAi en este caso)
chain = load_qa_chain(llm, chain_type="stuff", prompt=PROMPT)

# La pregunta que va a buscar la IA en los textos.
query = "¿A cuál clasificacion pertenece?"

# Busca todos fragmentos dentro de los documentos mas similares a la pregunta que se hizo.
docs = db.similarity_search(query)
context = '\n'.join([doc.page_content for doc in docs])
# print(context)

# Aca se ingresan los documentos relevantes encontrados en la sentencia anterior y ademas la query
respuesta_final = chain.run(input_documents=docs, question=query, context = context)
print("respuesta final", respuesta_final)







