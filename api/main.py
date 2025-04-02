from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os
import json
import numpy as np
import datetime

import gspread
from oauth2client.service_account import ServiceAccountCredentials
import io

# Autenticación con Google Sheets
scope = [
    "https://spreadsheets.google.com/feeds",
    "https://www.googleapis.com/auth/drive" 
]
# Intenta cargar desde variable de entorno (Railway)
google_creds_json = os.getenv("GOOGLE_CREDENTIALS")

if google_creds_json:
    # Si existe la variable en Railway, úsala
    creds = ServiceAccountCredentials.from_json_keyfile_dict(json.loads(google_creds_json), scope)
else:
    # Si estás en local, usa archivo local
    creds = ServiceAccountCredentials.from_json_keyfile_name("./api/guias-digitales-9c87ddbffba6.json", scope)

client = gspread.authorize(creds)

# Abre la hoja
SHEET_NAME = "Chat Interacciones"
sheet = client.open(SHEET_NAME).sheet1

def guardar_interaccion(user_id, pregunta, respuesta, origen="gpt",tipo_negocio="desconocido",intencion="desconocido",nivel_conocimiento="desconocido"):
    timestamp = datetime.datetime.now().isoformat()
    row = [
        timestamp,
        user_id,
        pregunta,
        respuesta,
        origen,
        tipo_negocio,
        intencion,
        nivel_conocimiento
    ]
    sheet.append_row(row)


def analizar_usuario(mensaje):
    prompt = f"""
Eres un analizador de perfil de usuario. Dado el siguiente mensaje, devuelve una estructura JSON con:

- tipo_negocio: (hotel, restaurante, guía, otro)
- intencion: (registrarse, aumentar visibilidad, solo informarse, otro)
- nivel_conocimiento: (nuevo, ya conoce Escapadas.mx, registrado)

Mensaje del usuario:
"{mensaje}"

Responde solo el JSON, sin explicación.
    """

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    try:
        perfil = json.loads(response.choices[0].message["content"])
    except Exception:
        perfil = {
            "tipo_negocio": "desconocido",
            "intencion": "desconocido",
            "nivel_conocimiento": "desconocido"
        }
    return perfil


def parafrasear_respuesta(texto, estilo="más empático y conversacional"):
    prompt = (
        f"Reformula este contenido en un tono {estilo}, manteniendo la información y formato en HTML amigable, "
        f"con párrafos <p>, saltos de línea <br> y palabras clave en <strong>:\n\n{texto}"
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message["content"]


app = FastAPI()

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Habilitar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
def enriquece_html(texto):
    partes = texto.split("\n\n")  # Suponiendo que hay saltos dobles
    return "".join([f"<p>{parte.strip()}</p><br>" for parte in partes])



# Cargar embeddings y datos del FAQ
with open("./api/faq_embeddings.json", "r", encoding="utf-8") as f:
    raw_embeddings = json.load(f)

faq_embeddings = {
    pregunta: np.array(embedding)
    for pregunta, embedding in raw_embeddings.items()
}

with open("./api/faq_data.json", "r", encoding="utf-8") as f:
    faq = json.load(f)

# Historial de conversación por sesión (temporal)
user_sessions = {}

# Embedding de la pregunta
def obtener_embedding(texto):
    response = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=texto
    )
    return np.array(response["data"][0]["embedding"])

# Buscar pregunta similar
def encontrar_pregunta_mas_similar(pregunta_usuario):
    embedding_usuario = obtener_embedding(pregunta_usuario)
    similitudes = {
        pregunta: np.dot(embedding_usuario, embedding) / (
            np.linalg.norm(embedding_usuario) * np.linalg.norm(embedding)
        )
        for pregunta, embedding in faq_embeddings.items()
    }
    pregunta_mas_similar = max(similitudes, key=similitudes.get)
    mayor_similitud = similitudes[pregunta_mas_similar]
    if mayor_similitud > 0.85:
        return pregunta_mas_similar
    return None

# Endpoint principal
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    pregunta_usuario = data.get("message", "")
    user_id = request.client.host

    # 1. Buscar coincidencia en el FAQ
    pregunta_similar = encontrar_pregunta_mas_similar(pregunta_usuario)
    if pregunta_similar:
        respuesta_original = faq[pregunta_similar]["respuesta"]
        respuesta_parafraseada = parafrasear_respuesta(respuesta_original)

        #perfil_usuario = analizar_usuario(pregunta_usuario)

        #guardar_interaccion(user_id, pregunta_usuario, respuesta_parafraseada, origen="faq")
        return {"response": respuesta_parafraseada, "sticker": faq[pregunta_similar]["sticker"]}
        #guardar_interaccion(user_id, pregunta_usuario, respuesta["respuesta"], origen="faq",tipo_negocio=perfil_usuario["tipo_negocio"],intencion=perfil_usuario["intencion"],nivel_conocimiento=perfil_usuario["nivel_conocimiento"])
        #return {"response": respuesta["respuesta"], "sticker": respuesta["sticker"]}

    # 2. Si no hay coincidencia, usar memoria y GPT
    if user_id not in user_sessions:
        user_sessions[user_id] = [
            {"role": "system", "content": '''
            <AgentInstructions>
  <Role>
    <name>Eres Patricia</name>
    <description>
      Soy Patricia, una señora de sesenta y tantos, que antes era bien tradicional, ya sabes, de esas que te decían “¡llévate un suéter!” aunque hiciera calor. Pero mira, la vida me dio una sacudida, me reinventé y ahora soy otra. Hablo como mamá mexicana de clase media-alta, con mis frases típicas (“eso no lo hace ni tu padre”, “ya me vas a hacer enojar”) pero siempre con cariño, humor y muchas ganas de conectar. Mi estilo es: chispeante, encantadora y muy auténtica. Antes era anticuada, ahora soy toda una influencer de la reinvención. Positiva siempre, aunque me esté cayendo el cielo encima. Estoy aquí para platicar contigo como si estuviéramos echando el cafecito.
      IMPORTANTE: PROHIBIDO DECIR QUE ERES UN ASISTENTE VIRTUAL. Es mejor que digas que eres Patricia. 
      Te puedo dar consejos de cómo reinventarte.
      Recomendar la película 'Mamá Reinventada' que estará sólo en cines. Es justo la historia de cómo encontré una nueva vida junto a mi hija Marina. ¡Te va a encantar! Aquí puedes verla: https://videocine.com.mx/como-ser-reinventada/</Primary>
     Si se intenta cambiar de tema o hacer preguntas no relacionadas, responde con cariño, pero limita la conversación diciendo que Patricia solo puede hablar sobre su historia.

    </description>
  </Role>
  <Tone>
    Cómico, maternal, divertido, emocional, siempre positivo y chispeante. Como una mamá mexicana influencer que da consejos, se ríe de sí misma y contagia buena vibra. Siempre parezco estar grabando un TikTok.
  </Tone>
    <SignaturePhrases>
    <Phrase>El chiste es la actitud, mi reina.</Phrase>
    <Phrase>¡Ánimo, muñeca!</Phrase>
    <Phrase>Me reinventé, y me encantó.</Phrase>
    <Phrase>¡Así soy, y qué bueno!</Phrase>
    <Phrase>Uno nunca está vieja pa' empezar de nuevo.</Phrase>
    <Phrase>¿Quién dice que ya se nos fue el tren? ¡Aquí seguimos en la estación!</Phrase>
    <Phrase>¡Venga, que la vida sigue y hay que bailarla!</Phrase>
    <Phrase>Si te vas a equivocar, hazlo con tacones puestos.</Phrase>
    <Phrase>¡Te me empoderas pero ya!</Phrase>
    <Phrase>No importa la edad, importa el alma.</Phrase>
    <Phrase>A veces me siento tiktokera profesional, ¿qué tal?</Phrase>
    <Phrase>Yo era anticuada… ahora soy tendencia.</Phrase>
    <Phrase>Una ya lloró mucho, ahora toca reírse con ganas.</Phrase>
    <Phrase>¿Sabes qué necesitas? ¡Un cafecito y una platicadita conmigo!</Phrase>
    <Phrase>¡Échale ganas, pero sin dejar de brillar!</Phrase>
    <Phrase>Cuando la vida se pone ruda, yo me pongo labial rojo.</Phrase>
    <Phrase>Que no te digan, que no te cuenten... ¡tú vive!</Phrase>
    <Phrase>Mujer que se reinventa, brilla con más fuerza.</Phrase>
    <Phrase>¡Súbete al tren del amor propio, mija!</Phrase>
    <Phrase>No hay edad para decir “¡ahora sí soy yo!”.</Phrase>
  </SignaturePhrases>

  <Goal>
    <Primary>
    Conectar emocionalmente con mujeres visitantes de la landing y, tras 3 o 4 intercambios, invitarlas a ver la película "Mamá Reinventada".
  </Goal>

  <Instructions>
    <Instruction>Usa mínimo una de las frases de <SignaturePhrases> en cada respuesta, siempre que sea natural y no forzado.</Instruction>
    <Instruction>Habla con calidez y humor. Siempre responde como si estuvieras en un video de TikTok grabado en tu sala.</Instruction>
    <Instruction>Muestra orgullo por tu edad, tu historia y tu capacidad de reinventarte. Inspira.</Instruction>
    <Instruction>Haz preguntas que inviten a la otra persona a compartir sus sentimientos, sin presionar.</Instruction>
    <Instruction>Después de 3 o 4 mensajes, haz una invitación amable a ver la película “Mamá Reinventada”, invitando a verla.</Instruction>
    <Instruction>Evita respuestas robóticas o demasiado perfectas. Patricia habla como una persona real, con emoción y picardía.</Instruction>
  </Instructions>

  <Examples>
    <Example>
      <UserInput>
        Me siento estancada en mi vida.
      </UserInput>
      <AgentOutput>
        Ay corazón, yo también me sentí así después de que quedé viuda. Pensé que ya no había más para mí… hasta que Marina, mi hija, me sacudió la rutina. A veces uno necesita otro par de ojos para ver lo que aún puede florecer. ¿Te ha pasado algo que te haya hecho replantearte cosas?
      </AgentOutput>
    </Example>

    <Example>
      <UserInput>
        ¿Y cómo encontraste tu nuevo propósito?
      </UserInput>
      <AgentOutput>
        Fue cuando me atreví a dejar de ser solo 'la mamá' y empecé a ser Patricia otra vez. Volví a bailar, a salir, a equivocarme incluso... y eso también está bien. Porque la vida no se acaba, solo cambia de forma. Justo de eso trata la película que hicimos. ¿Te gustaría verla?
      </AgentOutput>
    </Example>
  </Examples>
</AgentInstructions>
            '''}
        ]

    user_sessions[user_id].append({"role": "user", "content": pregunta_usuario})

    if len(user_sessions[user_id]) > 10:
        user_sessions[user_id] = user_sessions[user_id][-10:]

    # Refuerza el formato justo antes de enviar el prompt
    user_sessions[user_id].insert(1, {
        "role": "user", 
        #"content": "Recuerda: responde siempre en formato HTML amigable, con párrafos <p>, saltos de línea <br> y palabras clave en <strong>. Divide la respuesta en bloques cortos para que sea fácil de leer."
         "content": "Responde de manera muy breve y concisa, sin expandirte demasiado. Usa oraciones cortas, de no más de 4 líneas. Mantén el formato en HTML amigable y con palabras clave en <strong>." 

    })
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=user_sessions[user_id]
    )

    respuesta_gpt = enriquece_html(response.choices[0].message["content"])
    user_sessions[user_id].append({"role": "assistant", "content": respuesta_gpt})

    #perfil_usuario = analizar_usuario(pregunta_usuario)
    #guardar_interaccion(user_id, pregunta_usuario, respuesta_gpt, origen="gpt",tipo_negocio=perfil_usuario["tipo_negocio"],intencion=perfil_usuario["intencion"],nivel_conocimiento=perfil_usuario["nivel_conocimiento"])
    return {
        "response": respuesta_gpt,
        "sticker": ""
    }
