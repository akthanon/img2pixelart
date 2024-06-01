# img2pixelart INTERFAZ GRÁFICA
Codigo para transformar imagenes comunes a pixelart

# Requisitos
Pip (o en su defecto las librerías solicitadas). 

En caso de que no se tenga Pip, favor de instalar Curl o Wget.

- Python 3.x
- Bibliotecas Python: `cv2`, `numpy`, `PIL`, `scikit-learn`, `scipy`

Puedes instalar las bibliotecas necesarias usando pip:

pip install opencv-python numpy Pillow scikit-learn scipy

# Instalación
Clona este repositorio o solo descarga el update

# Opcional: Descarga la ultima versión estable del ejecutable.
- 🌟 [Img2pixelart](https://drive.google.com/file/d/1R4-WBI9fNJC2opKGBmyQYuHLAZpukXa3/view?usp=drive_link)

## Uso
Ejecuta la aplicación:

img2pixelart.bat

Selecciona la carpeta de entrada o la imagen de entrada.

Selecciona la carpeta de salida (opcional).

Elige una paleta de colores.

Configura el factor de escala o el tamaño de la imagen según tus preferencias.

Haz clic en "Procesar Imágenes" para convertir las imágenes.

# Opciones
Carpeta de Entrada: Puedes elegir una carpeta que contenga las imágenes que deseas convertir en arte pixelado.

Imagen de Entrada: Si prefieres, también puedes seleccionar una imagen específica.

Carpeta de Salida: Especifica la carpeta donde se guardarán las imágenes procesadas (opcional).

Paleta de Colores: Elige una paleta de colores para la conversión.

Indexar Colores: Puedes clusterizar tus imagenes indicando el número de colores que quieras (opción por defecto).

Factor de Escala / Tamaño de la Imagen: Decide si quieres ajustar el tamaño de las imágenes resultantes mediante el factor de escala o especificando directamente el tamaño de la imagen.
#
# img2pixelart CONSOLA DE COMANDOS

Este script convierte imágenes en arte de píxeles utilizando una paleta de colores predefinida. 

## Uso

### Opciones

El script acepta las siguientes opciones de línea de comandos:

- `-p, --palette_name`: Nombre de la paleta a utilizar (sin la extensión). Si no se proporciona, se utilizan los colores originales de la imagen.
- `-f, --scale_factor`: Factor de escala para la conversión a arte de píxeles.
- `-s, --image_size`: Tamaño de la imagen para la conversión a arte de píxeles (el valor más alto se utiliza conservando la relación de aspecto original).
- `-l, --list_palettes`: Lista todas las paletas disponibles.
- `-d, --input_folder`: Nombre del directorio de entrada que contiene las imágenes.
- `-o, --output_folder`: Nombre del directorio de salida donde se guardarán las imágenes convertidas.
- `-i, --input_image`: Nombre del archivo de imagen de entrada.
- `-n, --clusters_number`: Numero de clusters para indexar la imagen.
- `-di --dithering`: Dithering 0-100.
- `-de --denoise`: Denoise 0-100.

### Ejemplos de Uso

Para listar todas las paletas disponibles:

python script.py -l


Para convertir imágenes en un directorio a arte de píxeles:

python script.py -i input_image.jpg -o output_image -p palette_name -s image_size scikit-image


## Paletas Disponibles

Se proporcionan varias paletas de colores predefinidas en la carpeta `paletas`.

## Notas

- Si no se proporciona una paleta de colores, se utilizarán los colores originales de la imagen.
- Si se proporciona un factor de escala, se aplicará a la imagen para convertirla en arte de píxeles.
- Si se proporciona un tamaño de imagen, la imagen se redimensionará al tamaño especificado antes de convertirla en arte de píxeles.

¡Disfruta convirtiendo tus imágenes en arte de píxeles!


