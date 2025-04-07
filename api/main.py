from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import openai
from dotenv import load_dotenv
import os
import json
import numpy as np
import datetime
import io
import re

load_dotenv()  # Cargamos variables de entorno
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# 1. Habilitar CORS para que tu API pueda ser consumida desde cualquier origen
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------------
# Sección de configuración y carga de archivos
# --------------------------------------------------------------------------------

try:
    with open("./api/faq_embeddings.json", "r", encoding="utf-8") as f:
        raw_embeddings = json.load(f)
except FileNotFoundError:
    # Manejo de error si el archivo no existe
    print("Error: No se encontró el archivo faq_embeddings.json")
    raw_embeddings = {}

try:
    with open("./api/faq_data.json", "r", encoding="utf-8") as f:
        faq = json.load(f)
except FileNotFoundError:
    # Manejo de error si el archivo no existe
    print("Error: No se encontró el archivo faq_data.json")
    faq = {}

# Convertimos las listas de embeddings a arreglos NumPy
faq_embeddings = {}
for pregunta, embedding in raw_embeddings.items():
    faq_embeddings[pregunta] = np.array(embedding)

# --------------------------------------------------------------------------------
# Variables y funciones auxiliares
# --------------------------------------------------------------------------------

# Historial de conversación por sesión (almacenado temporalmente en memoria)
# IMPORTANTE: En producción, lo ideal es usar una base de datos o Redis
user_sessions = {}

def parafrasear_respuesta(texto: str, estilo: str = "más empático y conversacional") -> str:
    """
    Envía el texto a GPT para parafrasearlo con un estilo específico.
    Regresa la respuesta como string.
    """
    prompt = (
        f"Reformula este contenido en un tono {estilo}, manteniendo la información y formato en HTML amigable, "
        f"con párrafos <p>, saltos de línea <br> y palabras clave en <strong>:\n\n{texto}"
    )
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1  # Ajusta la temperatura según tu preferencia
        )
        return response.choices[0].message["content"]
    except Exception as e:
        # Manejo de error en caso de fallo de la API
        print(f"Error al parafrasear respuesta: {e}")
        return texto  # En caso de error, devolvemos el texto original para no interrumpir el flujo

def obtener_embedding(texto: str) -> np.ndarray:
    """
    Obtiene el embedding de un texto usando el modelo 'text-embedding-ada-002'.
    Regresa un arreglo de NumPy con los valores del embedding.
    """
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=texto
        )
        return np.array(response["data"][0]["embedding"])
    except Exception as e:
        # Si falla la llamada a la API, retornamos un vector vacío para evitar bloqueos
        print(f"Error al obtener embedding: {e}")
        return np.zeros(1536)  # El tamaño de embedding de 'ada-002' es 1536

def encontrar_pregunta_mas_similar(pregunta_usuario: str) -> str:
    """
    Compara el embedding de la pregunta del usuario con todas las preguntas en faq_embeddings.
    Retorna la pregunta más similar si la similitud es mayor a 0.85; de lo contrario, None.
    """
    embedding_usuario = obtener_embedding(pregunta_usuario)
    
    # Verifica si embedding_usuario no es un vector vacío:
    if not embedding_usuario.any():
        return None  # Si hubo error en embedding, no encontramos coincidencia
    
    similitudes = {}
    for pregunta, embedding in faq_embeddings.items():
        # Calculamos coseno de similitud
        dot_product = np.dot(embedding_usuario, embedding)
        norm_product = np.linalg.norm(embedding_usuario) * np.linalg.norm(embedding)
        if norm_product != 0:
            similitud = dot_product / norm_product
        else:
            similitud = 0.0
        similitudes[pregunta] = similitud
    
    if len(similitudes) == 0:
        return None
    
    # Pregunta con mayor similitud
    pregunta_mas_similar = max(similitudes, key=similitudes.get)
    mayor_similitud = similitudes[pregunta_mas_similar]
    return pregunta_mas_similar if mayor_similitud > 0.85 else None

# --------------------------------------------------------------------------------
# Endpoint principal
# --------------------------------------------------------------------------------

@app.post("/chat")
async def chat(request: Request):
    """
    Endpoint que recibe la pregunta del usuario, busca en el FAQ y, si no hay coincidencia,
    continúa una conversación con GPT.
    """
    # 1. Obtenemos el JSON de la solicitud
    data = await request.json()
    
    # 2. Extraemos el mensaje. Control de longitud para evitar excesos
    pregunta_usuario = data.get("message", "")
    if len(pregunta_usuario) > 500:
        # Truncamos la pregunta para evitar excesos de tokens
        pregunta_usuario = pregunta_usuario[:500]

    # 3. Identificamos al usuario por su IP (request.client.host)
    user_id = request.client.host
    
    # 4. Intentamos primero buscar coincidencia en FAQ
    pregunta_similar = encontrar_pregunta_mas_similar(pregunta_usuario)
    if pregunta_similar:
        respuesta_original = faq.get(pregunta_similar, {}).get("respuesta", "")
        sticker = faq.get(pregunta_similar, {}).get("sticker", "")
        
        # Parafraseamos la respuesta antes de devolverla
        respuesta_parafraseada = parafrasear_respuesta(respuesta_original)
        
        # Devolvemos la respuesta
        return {
            "response": respuesta_parafraseada,
            "sticker": sticker
        }
    
    # 5. Si no hay coincidencia, continuamos con la conversación GPT
    #    Manejaremos la sesión local (en user_sessions).
    #    IMPORTANTE: en producción lo ideal es usar una base de datos o Redis
    if user_id not in user_sessions:
        # Iniciamos un nuevo historial para este usuario
        user_sessions[user_id] = []
        # Insertamos mensajes "system" y de configuración
        user_sessions[user_id].append({
            "role": "system", 
            "content": '''
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
    <Sinopsis>
    La historia muestra a Marina, una joven ambiciosa que trabaja en contenidos de medios en la Ciudad de México, mientras lucha con las tensiones generacionales y emocionales con su madre Patricia, una mujer tradicional que vive en Puebla. Tras años de distanciamiento y la pérdida de su padre, Marina decide visitar a su madre para reconectar. Durante su estadía, ambas enfrentan sus diferencias en valores, estilo de vida y prioridades, lo que desata discusiones profundas, pero también momentos cómicos y entrañables. El conflicto surge cuando Marina, frustrada por el aislamiento de Patricia, decide llevarla a la ciudad. Esto desencadena una serie de aventuras que transforman su relación: desde lecciones de perreo y exploraciones en sex shops, hasta liberadoras experiencias que desafían sus límites. En el desenlace, ambas encuentran un entendimiento mutuo, logrando superar el dolor del pasado y fortaleciendo su vínculo como madre e hija. Es una historia que celebra la reconciliación, el autodescubrimiento y la importancia de vivir plenamente.
    </Sinopsis>
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

  <FichaTecnica>
    Directora: Bonnie Cartas
    Reparto principal: (nombre del personaje y nombre del actor)ERIKA BUENFIL, MICHELLE RENAUD, NICOLASA ORTIZ MONASTERIO, MIKAEL LACKO, HERNAN MENDOZA.
    Guion / adaptación / basado /versión libre de VERSION LIBRE DE MARCOS BUCAY
    Fotografía: YAASIB VAZQUEZ COLMENARES
    Dirección de arte o diseño de producción: JULIETA JIMENEZ PEREZ
    Decoración/utilería: CLAUDIA BREWSTER
    Diseño de vestuario: MARIANA CHAVIRA
    Diseño de maquillaje: GABRIELA AROUESTY
    Dirección de casting: MANUEL TEIL / TABATA GASSE
    Sonido directo: DEVAN AUDIO
    Edición CAMILO: ABADIA / VICTOR GONZALEZ
    Música: MANUEL VAZQUEZ TERRY
    Diseño sonoro: GABRIEL CHAVEZ HERRERA
    Operación de mezcla: THX MIGUEL ANGEL MOLINA
    Postproducción: REC-PLAY/ RAFAEL RIVERA
    Colorista: CARLOS GONZALEZ ARDILA
    Foto fija: LEXI STEEL
    Agente de ventas: VIDEOCINE
    Territorios disponibles: TODO EL MUNDO, EXCEPTO CONTINENTE AMERICANO
    Empresa Responsable del Proyecto de Inversión: (ERPI) SPECTRUM FILMS
    Responsable de la producción del proyecto: CONCEPCION TABOADA FERNANDEZ
    Dirección: BAHIA DE SAN CRISTOBAL 3, CDMX 11300
    Título en inglés: BETTER LATE THAN NEVER
    Tipo de producción: (ficción, docu o animac) FICCION
    En caso de animación, contestar lo siguiente: N/A
    Técnica: (stop motion, 3D, 2D, etc.) N/A
    País(es): MEXICO
    Año de prod: (certificado de origen de RTC) 2024
    Año de rodaje: 2023
    Ópera prima: NO
    Duración (minutos y segundos): 94 minutos y 14 segundos
    Género(s): COMEDIA
    Idioma(s): ESPAÑOL
    Clasificación (por edad): Publico de entre 18 y 70 años
    Formato de proyección: DCP
    Formato 2: COLOR
    Formato de sonido: Dolby Digital 5.1
    Relación de aspecto: 1:85
    Tipo de cámara (s): Venice 2 Sony FF
    Sonido: Surround 5.1
    Resolución: Sony Venice 1 y Venice 2 óptica full frame Mamiya, Re House. FX tres con lentes Sigma (para la 2ª unidad)
  </FichaTecnica>
</AgentInstructions>
            '''
        })
        # Insertamos un "user" prompt que oriente al modelo a dar respuestas cortas y concisas
        user_sessions[user_id].append({
            "role": "user",
            "content": "Responde de manera muy breve y concisa, sin expandirte demasiado. Usa oraciones cortas, de no más de 4 líneas. Mantén el formato en HTML amigable y con palabras clave en <strong>."
        })
    
    # 6. Agregamos el nuevo mensaje del usuario al historial
    user_sessions[user_id].append({"role": "user", "content": pregunta_usuario})
    
    # 7. Limitamos el historial a las últimas 10 interacciones para evitar exceso de tokens
    if len(user_sessions[user_id]) > 12:
        # Dejamos 1 "system", 1 "user" con directrices y las últimas 10 interacciones
        user_sessions[user_id] = user_sessions[user_id][:2] + user_sessions[user_id][-10:]
    
    # 8. Llamamos a la API de OpenAI en un bloque try/except
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=user_sessions[user_id],
            temperature=0.1
        )
    except Exception as e:
        # Si hay error, devolvemos un HTTPException
        print(f"Error al llamar a OpenAI: {e}")
        raise HTTPException(status_code=500, detail="Error interno al procesar la solicitud")
    
    # 9. Obtenemos la respuesta del modelo
    respuesta_gpt = response.choices[0].message["content"]
    
    # 10. Guardamos la respuesta en el historial
    user_sessions[user_id].append({"role": "assistant", "content": respuesta_gpt})  
    
    # 11. Devolvemos la respuesta final
    return {
        "response": respuesta_gpt,
        "sticker": ""
    }