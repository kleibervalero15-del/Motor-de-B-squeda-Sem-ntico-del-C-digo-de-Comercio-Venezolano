import streamlit as st
import re
import os
from pypdf import PdfReader # Librería para la extracción del PDF
from sentence_transformers import SentenceTransformer # IA de embeddings
from sklearn.metrics.pairwise import cosine_similarity # Para medir la similitud semántica
from typing import Dict, List, Tuple

# =========================================================================
# 1. FUNCIÓN DE CARGA DEL CORPUS (Extrae todos los artículos de su PDF)
# =========================================================================

# NOTA IMPORTANTE: Usted debe colocar su archivo 'codigo de comercio.pdf' 
# en el mismo directorio que este script de Python.

@st.cache_resource # Cachea el resultado para no cargar el PDF y el modelo en cada interacción
def load_corpus_from_pdf(pdf_path: str) -> Dict[str, str]:
    """
    Función profesional para extraer y segmentar todos los artículos 
    del Código de Comercio desde el PDF.
    """
    if not os.path.exists(pdf_path):
        st.error(f"Error: El archivo '{pdf_path}' no se encuentra. Asegúrese de colocar su PDF en el mismo directorio.")
        return {}
    
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"
    
    # Expresión Regular para segmentar por Artículo N°
    # Busca 'Artículo ' seguido de cualquier número (ej. 1°, 354°, 1.120°)
    # Esto es crucial para separar cada disposición legal.
    articles = re.split(r'(Artículo \d+\.?\d*°)', full_text)
    
    corpus = {}
    current_title = ""
    for segment in articles:
        segment = segment.strip()
        if segment.startswith("Artículo"):
            current_title = segment
        elif current_title and segment:
            # Limpieza básica para eliminar saltos de línea innecesarios
            cleaned_text = re.sub(r'\s+', ' ', segment).strip()
            corpus[current_title] = cleaned_text
            current_title = "" # Reiniciar título
            
    # Filtro final: Asegurarse de que el corpus no esté vacío y tiene contenido relevante
    final_corpus = {k: v for k, v in corpus.items() if len(v) > 20}
    return final_corpus

# =========================================================================
# 2. IA INTEGRADA: MOTOR DE BÚSQUEDA SEMÁNTICA (El corazón del 'Google')
# =========================================================================

# Carga del modelo de IA (sólo se hace una vez)
@st.cache_resource
def load_ia_model():
    """Carga un modelo de embeddings pre-entrenado para el idioma español."""
    # ¡MODELO CORREGIDO PARA ESPAÑOL!
    return SentenceTransformer('hiiamsid/sentence_similarity_spanish_es') 
    # Alternativa profesional y muy robusta: 'paraphrase-multilingual-mpnet-base-v2'
    
def semantic_search(query: str, corpus: Dict[str, str], model, corpus_embeddings) -> List[Tuple[str, str]]:
    """
    Realiza una búsqueda semántica de alta precisión usando la Similitud del Coseno.
    """
    
    # 1. Vectorización de la consulta del usuario
    query_embedding = model.encode(query, convert_to_tensor=True).cpu().numpy().reshape(1, -1)
    
    # 2. Cálculo de la similitud del coseno entre la consulta y todos los artículos
    # Esto es el núcleo de la IA: encuentra el 'significado' más cercano, no solo la palabra.
    similarities = cosine_similarity(query_embedding, corpus_embeddings)[0]
    
    # 3. Empaquetar resultados con sus puntajes
    results_with_scores = []
    titles = list(corpus.keys())
    for i, score in enumerate(similarities):
        results_with_scores.append((score, titles[i], corpus[titles[i]]))
        
    # 4. Ordenar por puntaje (relevancia)
    results_with_scores.sort(key=lambda x: x[0], reverse=True)
    
    # Devolver los 5 mejores resultados
    return [(titulo, texto) for score, titulo, texto in results_with_scores[:5]]

# =========================================================================
# 3. INTERFAZ GRÁFICA (Streamlit)
# =========================================================================

# --- Configuración Estética ---
st.set_page_config(
    page_title="LexMercantil IA - Código de Comercio VE",
    page_icon="🇻🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS
st.markdown("""
    <style>
    .big-title { font-size: 48px !important; font-weight: 700; color: #000080; /* Azul oscuro */ text-align: center; margin-bottom: 0px; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .st-emotion-cache-1c9as99, .st-emotion-cache-162n78c { border-radius: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); padding: 25px; border-left: 6px solid #FF4B4B; /* Rojo */ }
    .reportview-container .main { background-color: #f8f9fa; /* Fondo muy suave */ }
    </style>
    """, unsafe_allow_html=True)


# --- Proceso de Carga y Cacheo ---

# 1. Cargar el modelo de IA
with st.spinner('Cargando Modelo de Inteligencia Artificial (NLP)...'):
    try:
        model_ia = load_ia_model()
    except Exception as e:
        st.error(f"Error al cargar el modelo de IA. ¿Instaló 'sentence-transformers'? Error: {e}")
        st.stop()

# 2. Cargar el corpus completo desde el PDF
with st.spinner('Extrayendo y Segmentando TODOS los Artículos del Código de Comercio...'):
    CORPUS_CODIGO_COMERCIO = load_corpus_from_pdf("codigo de comercio.pdf")

if not CORPUS_CODIGO_COMERCIO:
    st.error("No se pudo cargar el corpus. Por favor, corrija los errores indicados arriba.")
    st.stop()
    
# 3. Vectorizar el Corpus (La IA lee los artículos)
@st.cache_resource
def get_corpus_embeddings(corpus,_model):
    st.info(f"Vectorizando {len(corpus)} artículos para la búsqueda semántica...")
    return _model.encode(list(corpus.values()), convert_to_tensor=True).cpu().numpy()

with st.spinner(f'Analizando semánticamente {len(CORPUS_CODIGO_COMERCIO)} Artículos...'):
    CORPUS_EMBEDDINGS = get_corpus_embeddings(CORPUS_CODIGO_COMERCIO, model_ia)

# --- Interfaz Principal ---

st.markdown('<p class="big-title">🇻🇪 LexMercantil IA</p>', unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: #555;'>Motor de Búsqueda Semántico del Código de Comercio Venezolano</h3>", unsafe_allow_html=True)
st.divider()

col_main = st.columns([1, 4, 1])[1] 

with col_main:
    search_query = st.text_input(
        "Su consulta:",
        placeholder="Ej. '¿Qué sucede si una mujer casada quiere ser comerciante?'",
        label_visibility="collapsed"
    )

    if st.button("🔎 Búsqueda Semántica con IA", use_container_width=True, type="primary"):
        if search_query:
            with st.spinner('Buscando significado y contexto en el Código...'):
                # Llamada al núcleo de búsqueda IA (Semantic Search)
                resultados = semantic_search(search_query, CORPUS_CODIGO_COMERCIO, model_ia, CORPUS_EMBEDDINGS)
            
            st.divider()
            
            if resultados:
                st.success(f"Resultados más relevantes (IA) para: **'{search_query}'**")
                
                for titulo, texto in resultados:
                    st.info(f"**{titulo}**", icon="📄")
                    st.markdown(f"<p style='padding-left: 20px;'>{texto}</p>", unsafe_allow_html=True)
                    st.caption("---")
            else:
                st.warning("No se encontraron resultados relevantes. Intente una consulta más específica.")
        else:
            st.error("Por favor, introduzca un término de búsqueda.")

# --- Sidebar Profesional ---
with st.sidebar:
    st.markdown("## ⚙️ Detalles Técnicos del Sistema")
    st.metric(label="Artículos Cargados (Corpus)", value=len(CORPUS_CODIGO_COMERCIO))
    st.metric(label="Tipo de Búsqueda", value="IA Semántica (Embeddings)")
    st.markdown("---")
    st.markdown("""
    Este sistema ya es el **"Google del Código de Comercio"**. 
    * Utiliza `pypdf` para extraer la totalidad de los 1.120+ artículos.
    * Emplea `sentence-transformers` para vectorizar (convertir a significado numérico) la consulta.
    * Los resultados son los artículos cuyo **significado** es más cercano al de la consulta (Similitud del Coseno).
    """)