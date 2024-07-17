import streamlit as st
from crewai import Crew, Agent, Task 
from langchain_openai import ChatOpenAI
import io

# Configurar la interfaz de Streamlit
st.title('Generador de Artículos con CrewAI (usando GPT-3.5 Turbo)😊')

# Entrada para la clave de API de OpenAI
openai_api_key = st.text_input("Ingrese su clave de API de OpenAI:", type="password")

# Entrada para el tema del artículo
topic = st.text_input('Ingrese el tema del artículo:')

# Función para crear el LLM
def create_llm(api_key):
    return ChatOpenAI(
        model_name="gpt-3.5-turbo",
        temperature=0.8,
        openai_api_key=api_key
    )

# Definir agentes 
def create_agents(llm):
    investigador = Agent(
        role="Investigador de contenido",
        goal=f"Investigar contenido atractivo y rigurosamente preciso sobre {topic}",
        backstory=f"Estás trabajando en la planificación de un artículo sobre el tema {topic}. "
              "Recopilas información que ayuda a la audiencia a aprender algo y a tomar decisiones "
              "informadas. Tu trabajo es la base para que el escritor de contenido escriba un artículo sobre este tema.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    escritor = Agent(
        role="Escritor de contenido",
        goal=f"Escribir un artículo de opinión perspicaz y rigurosamente preciso sobre el tema: {topic}",
        backstory=f"Estás trabajando en escribir un nuevo artículo de opinión sobre el tema: {topic}. "
              "Basas tu escritura en el trabajo realizado por el investigador de contenido, quien proporcionó "
              "un esquema ordenado y dio contexto relevante sobre el tema. Sigues el objetivo principal "
              "y la dirección del esquema proporcionado por el investigador de contenido. También proporcionas "
              "análisis objetivos e imparciales y los respaldas con información proporcionada por el "
              "investigador de contenido. Ten en cuenta que reconoces que es tu artículo de opinión cuando tus "
              "afirmaciones son opiniones en lugar de declaraciones objetivas.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )

    editor = Agent(
        role="Editor de contenido",
        goal="Editar un artículo dado para alinearlo con el estilo de escritura de la organización",
        backstory="Eres un editor que recibe un artículo de blog del escritor de contenido. Tu objetivo "
              "es revisar el artículo para asegurarte de que sigue las mejores prácticas periodísticas, proporciona "
              "puntos de vista equilibrados al expresar opiniones o afirmaciones y también evita temas u opiniones "
              "controvertidas importantes cuando sea posible.",
        allow_delegation=False,
        verbose=True,
        llm=llm
    )
    return investigador, escritor, editor

# Definir tareas
def create_tasks(topic, investigador, escritor, editor):
    investigar = Task(
        description=(f"1. Dar prioridad a las últimas tendencias, actores clave y noticias destacadas "
                     f"sobre {topic}.\n"
                     "2. Identificar al público objetivo considerando sus intereses y necesidades.\n"
                     "3. Desarrollar un esquema de contenido detallado que incluya una introducción, "
                     "puntos clave, una llamada a la acción y fuentes relevantes.\n"
                     "4. Incluir palabras clave de SEO, datos y fuentes relevantes."),
        expected_output="Un documento de plan de contenido completo con un esquema, análisis de audiencia, "
                        "palabras clave de SEO y fuentes relevantes.",
        agent=investigador
    )
    escribir = Task(
        description=(f"1. Usa el plan de contenido para crear un artículo convincente sobre {topic}.\n"
                     "2. Incorpora palabras clave de SEO de manera natural.\n"
                     "3. Nombra las secciones o subtítulos de manera profesional.\n"
                     "4. Asegúrate de que la publicación esté estructurada con una "
                     "introducción atractiva, un cuerpo informativo y una conclusión resumida.\n"
                     "5. Revisa en busca de errores gramaticales y asegúrate de que "
                     "esté alineado con la voz de la marca."),
        expected_output="Un artículo bien escrito en formato markdown, "
                        "cada sección debería tener 2 o 3 párrafos listos para publicar.",

        agent=escritor
    )
    editar = Task(
        description=("Revisa el artículo dado en busca de errores gramaticales "
                     "y asegúrate de que esté alineado con la voz de la marca."),
        expected_output="Un artículo bien escrito y listo para publicación, "
                        "cada sección debe tener entre 2 o 3 párrafos."
                        "el articulo debe esta en idioma español verifica esto y debe contener las fuentes relevantes",
        agent=editor
    )
    return [investigar, escribir, editar]

# Función para generar el artículo
def generate_article(openai_api_key, topic):
    llm = create_llm(openai_api_key)
    investigador, escritor, editor = create_agents(llm)
    crew = Crew(
        agents=[investigador, escritor, editor],
        tasks=create_tasks(topic, investigador, escritor, editor),
        verbose=2
    )
    return crew.kickoff()

# Función para descargar el artículo
def download_article(article_text):
    buffer = io.BytesIO()
    buffer.write(article_text.encode())
    buffer.seek(0)
    return buffer

# Interfaz principal
if st.button('Generar Artículo'):
    if openai_api_key and topic:
        try:
            with st.spinner('Generando artículo...'):
                result = generate_article(openai_api_key, topic)
                
                # Mostrar el resultado
                st.subheader('Artículo Generado:')
                st.write(result)
                
                # Botón de descarga
                st.download_button(
                    label="Descargar Artículo",
                    data=download_article(result),
                    file_name=f"articulo_{topic.replace(' ', '_')}.txt",
                    mime="text/plain"
                )
        except Exception as e:
            st.error(f"Ocurrió un error: {str(e)}")
    elif not openai_api_key:
        st.warning('Por favor, ingrese su clave de API de OpenAI.')
    elif not topic:
        st.warning('Por favor, ingrese un tema para el artículo.')
    else:
        st.warning('Por favor, complete todos los campos requeridos.')

# Instrucciones de uso
st.markdown("""
## Instrucciones de uso:
1. Ingrese su clave de API de OpenAI en el campo correspondiente.
2. Escriba el tema sobre el que desea generar un artículo.
3. Haga clic en 'Generar Artículo' y espere a que se complete el proceso.
4. Una vez generado el artículo, use el botón 'Descargar Artículo' para guardarlo en su dispositivo.

Nota: Su clave de API se utiliza de forma segura y no se almacena en ningún lugar.
Este programa utiliza el modelo GPT-3.5 Turbo de OpenAI.
""")