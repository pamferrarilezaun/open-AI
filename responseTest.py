import openai

# Configura tu clave de API
openai.api_key = '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'

# Realiza una llamada al modelo de chat
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Who won the world series in 2020?"},
        {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
        {"role": "user", "content": "Where was it played?"}
    ]
)

# Verifica si se obtuvo una respuesta válida
if 'choices' in response and len(response.choices) > 0:
    message = response.choices[0].message
    if 'role' in message and message['role'] == 'assistant':
        print("¡El modelo ha respondido correctamente!")
        print("Respuesta:", message['content'])
    else:
        print("No se obtuvo una respuesta del asistente")
else:
    print("No se obtuvo respuesta del modelo")
