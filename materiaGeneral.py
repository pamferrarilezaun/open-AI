from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain import OpenAI, VectorDBQA
from langchain.document_loaders import DirectoryLoader
import magic
import os
import nltk

os.environ['OPENAI_API_KEY'] = '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'

# nltk.download('averaged_perceptron_tagger')

# Esto me sirvio para instalar las librerias.
# pip install unstructured
# Other dependencies to install https://langchain.readthedocs.io/en/latest/modules/document_loaders/examples/unstructured_file.html
# pip install python-magic-bin
# pip install chromadb

# Se obtienen todos los archivos txt
loader = DirectoryLoader('data/', glob='**/*.txt')
print(loader)
documents = loader.load()


# Divide el texto de entrada en partes significativas para poder procesarlo.
# Se divide en fragmentos con un tamanio de 1000 y el segundo parametro es para que no superpongan los fragmentos
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

# Se corte cada texto de entrada por documento
texts = text_splitter.split_documents(documents)

# print(texts)

#Esto convierte de una cadena de caracteres a un espacio vectorial. Basicamente convierte palabras
# a una lista valores numericos. 
# Se utiliza chroma para crear un documento o un conjunto de vectores sobre el texto que se pasa en el primer
# parametro y sobre el motor que esta embebido (pasado en el segundo parametro) 
# Se realiza una consulta a la API de openAI
embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
docsearch = Chroma.from_documents(texts, embeddings)

# En esta linea se inicializa nuestro modelo: se le pasa nuestro modelo de lenguaje (primer param.),
# el segundo parametro 'Chain type of staff' significa que vamos a encontrar los fragmentos mas relevantes del texto
# de todas las cadenas de texto que realizamos un split en la linea 28 
# El tercer parametro es el vector de chroma
qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch)

query = "Hace un resumen juridico de los textos"

respuesta = qa.run(query)
print(respuesta)

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
query = "¿De que hablan los textos en su conjunto?"
result = qa({"query": query})
print(result['result'])

qa = VectorDBQA.from_chain_type(llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True)
query = "Realiza un sumario juridico de cada una de los textos"
result = qa({"query": query})
print("sumario juridico:", result['result'])



