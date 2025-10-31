
import os
import pandas as pd


def load_divorce_data(data_path="01-divorce-eda/data"):
    """
    Carga el dataset de Predictores de Divorcio desde UCI.
    
    El dataset contiene 170 instancias con 54 atributos basados en la
    Escala de Medición de Divorcio de Gottman, más una columna objetivo
    que indica si la pareja está divorciada (1) o casada (0).
    
    Args:
        data_path: Ruta base para almacenar los datos
        
    Returns:
        DataFrame de pandas con los datos cargados
    """
    # Crear directorios si no existen
    raw_path = os.path.join(data_path, "raw")
    processed_path = os.path.join(data_path, "processed")
    os.makedirs(raw_path, exist_ok=True)
    os.makedirs(processed_path, exist_ok=True)
    
    # Archivo local en el directorio del proyecto
    filename = os.path.join(raw_path, "divorce.csv")
    
    # Verificar si ya existe en raw/
    if os.path.exists(filename):
        print(f"ℹ️  El archivo {filename} ya existe, usando versión local")
        df = pd.read_csv(filename, sep=';', encoding='utf-8', on_bad_lines='skip')
    else:
        # Archivo fuente en la raíz del repositorio
        # Obtener la ruta absoluta del script actual
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Subir dos niveles: scripts -> 01-divorce-eda -> MLO
        repo_root = os.path.join(script_dir, "..", "..")
        source_file = os.path.join(repo_root, "divorce.csv")
        source_file = os.path.abspath(source_file)
        
        if os.path.exists(source_file):
            # Copiar desde la raíz del repositorio
            print(f"ℹ️  Copiando archivo desde {source_file}")
            # Leer el archivo con manejo de líneas mal formadas
            df = pd.read_csv(source_file, sep=';', encoding='utf-8', on_bad_lines='skip')
            # Guardar en raw/
            df.to_csv(filename, index=False, sep=';')
            print(f"✅ Archivo copiado a: {filename}")
        else:
            # Archivo no encontrado
            raise FileNotFoundError(
                f"No se encontró el archivo de datos.\n"
                f"Buscado en:\n"
                f"  - {filename}\n"
                f"  - {source_file}\n\n"
                f"Por favor, descarga el archivo manualmente desde:\n"
                f"https://archive.ics.uci.edu/dataset/539/divorce+predictors+data+set\n"
                f"Y guárdalo como: {source_file}"
            )
    
    # Información básica
    print(f"\n📊 Datos cargados exitosamente!")
    print(f"   - Forma: {df.shape}")
    print(f"   - Columnas: {df.shape[1]}")
    print(f"   - Filas: {df.shape[0]}")
    
    # Renombrar columnas para mejor legibilidad si es necesario
    # La última columna es el objetivo (Class o Divorce)
    if 'Class' in df.columns:
        df = df.rename(columns={'Class': 'Divorce'})
    
    # Si las columnas no están nombradas como Atr1, Atr2, etc., renombrarlas
    if not all(col.startswith('Atr') or col == 'Divorce' for col in df.columns):
        feature_names = [f"Atr{i+1}" for i in range(df.shape[1] - 1)]
        feature_names.append("Divorce")
        df.columns = feature_names
    
    # Guardar versión procesada
    processed_file = os.path.join(processed_path, "divorce_processed.csv")
    df.to_csv(processed_file, index=False)
    print(f"💾 Datos procesados guardados en: {processed_file}")
    
    return df


def get_feature_descriptions():
    """
    Retorna un diccionario con las descripciones de las características del dataset.
    
    Las preguntas están basadas en la Escala de Medición de Divorcio de Gottman
    y se califican en escala Likert de 0 (Nunca) a 4 (Siempre).
    """
    descriptions = {
        "Atr1": "Si uno de nosotros se disculpa cuando la discusión es exagerada, la discusión termina.",
        "Atr2": "Sé que podemos ignorar nuestras diferencias, incluso si las cosas se ponen difíciles a veces.",
        "Atr3": "Cuando necesitamos, podemos tomar nuestras discusiones con mi cónyuge desde el principio y corregirlas.",
        "Atr4": "Cuando discuto con mi cónyuge, eventualmente me contactaré con él.",
        "Atr5": "El tiempo que paso con mi esposa es especial para nosotros.",
        "Atr6": "No tenemos tiempo en casa como pareja.",
        "Atr7": "Somos como dos extraños que comparten el mismo entorno en casa en lugar de familia.",
        "Atr8": "Disfruto nuestras vacaciones con mi esposa.",
        "Atr9": "Disfruto viajar con mi esposa.",
        "Atr10": "La mayoría de nuestros objetivos son comunes para mi cónyuge y para mí.",
        "Atr11": "Creo que algún día en el futuro, cuando mire hacia atrás, veré que mi cónyuge y yo hemos estado en armonía entre sí.",
        "Atr12": "Mi cónyuge y yo tenemos valores similares en términos de libertad personal.",
        "Atr13": "Mi cónyuge y yo tenemos el mismo sentido de entretenimiento.",
        "Atr14": "La mayoría de nuestros objetivos para las personas (niños, amigos, etc.) son los mismos.",
        "Atr15": "Nuestros sueños con mi cónyuge son similares y armoniosos.",
        "Atr16": "Somos compatibles con mi cónyuge sobre lo que debe ser el amor.",
        "Atr17": "Compartimos los mismos puntos de vista con mi cónyuge sobre ser feliz en nuestra vida.",
        "Atr18": "Mi cónyuge y yo tenemos ideas similares sobre cómo debe ser el matrimonio.",
        "Atr19": "Mi cónyuge y yo tenemos ideas similares sobre cómo deben ser los roles en el matrimonio.",
        "Atr20": "Mi cónyuge y yo tenemos valores similares en confianza.",
        "Atr21": "Sé exactamente lo que mi cónyuge quiere decir.",
        "Atr22": "Sabemos lo que debería ser el día de cada uno (en casa y en el trabajo).",
        "Atr23": "Conozco bien a mi cónyuge.",
        "Atr24": "Sé cómo mi cónyuge quiere que lo cuiden cuando está enfermo.",
        "Atr25": "Conozco el mundo interior de mi cónyuge.",
        "Atr26": "Conozco los miedos básicos de mi cónyuge.",
        "Atr27": "Sé cuáles son las fuentes de estrés de mi cónyuge en su vida.",
        "Atr28": "Conozco las esperanzas y deseos de mi cónyuge.",
        "Atr29": "Conozco muy bien a mi cónyuge.",
        "Atr30": "Conozco a los amigos de mi cónyuge y sus relaciones sociales.",
        "Atr31": "Me siento agresivo cuando discuto con mi cónyuge.",
        "Atr32": "Cuando discuto con mi cónyuge, generalmente uso expresiones como 'tú siempre' o 'tú nunca'.",
        "Atr33": "Puedo usar declaraciones negativas sobre la personalidad de mi cónyuge durante nuestras discusiones.",
        "Atr34": "Puedo usar expresiones ofensivas durante nuestras discusiones.",
        "Atr35": "Puedo insultar a mi cónyuge durante nuestras discusiones.",
        "Atr36": "Puedo ser humillante cuando discutimos.",
        "Atr37": "Mi discusión con mi cónyuge no es tranquila.",
        "Atr38": "Odio la forma en que mi cónyuge abre un tema.",
        "Atr39": "Nuestras discusiones a menudo ocurren repentinamente.",
        "Atr40": "Simplemente nos calmamos un poco y luego discutimos nuevamente sobre el tema.",
        "Atr41": "Cuando hablo con mi cónyuge sobre algo, mi calma se rompe repentinamente.",
        "Atr42": "Cuando discuto con mi cónyuge, solo me quedo en silencio.",
        "Atr43": "A veces pienso que es bueno para mí irme de casa por un tiempo.",
        "Atr44": "Preferiría quedarme en silencio que discutir con mi cónyuge.",
        "Atr45": "Incluso si estoy en lo correcto en la discusión, me quedo en silencio para no lastimar a mi cónyuge.",
        "Atr46": "Cuando discuto con mi cónyuge, me quedo en silencio porque tengo miedo de no poder controlar mi ira.",
        "Atr47": "Siento que estoy bien en nuestras discusiones.",
        "Atr48": "No tengo nada que ver con lo que he sido acusado.",
        "Atr49": "En realidad no soy el responsable de lo que me acusan.",
        "Atr50": "No soy el que está equivocado en nuestras discusiones.",
        "Atr51": "No dudaría en decirle a mi cónyuge sobre su insuficiencia.",
        "Atr52": "Cuando discuto, le recuerdo a mi cónyuge sus defectos.",
        "Atr53": "No tengo miedo de decirle a mi cónyuge sobre su incompetencia.",
        "Atr54": "El comportamiento negativo de mi cónyuge me hace recordar sus defectos.",
        "Divorce": "Variable objetivo: 1 = Divorciado, 0 = Casado"
    }
    return descriptions


if __name__ == '__main__':
    # Cargar datos
    df = load_divorce_data()
    
    # Mostrar información básica
    print("\n" + "="*60)
    print("INFORMACIÓN DEL DATASET")
    print("="*60)
    print(f"\nPrimeras filas:")
    print(df.head())
    
    print(f"\nEstadísticas básicas:")
    print(df.describe())
    
    print(f"\nDistribución de la variable objetivo:")
    print(df['Divorce'].value_counts())
    print(f"\nPorcentaje de divorciados: {df['Divorce'].mean()*100:.2f}%")
