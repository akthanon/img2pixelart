import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageStat
import os
import argparse
from sklearn.cluster import KMeans
from scipy.spatial import KDTree

def denoise_image(img):
    img_np = np.array(img.convert('RGBA'))
    rgb_np = img_np[..., :3]
    alpha_np = img_np[..., 3]
    denoised_rgb_np = cv2.bilateralFilter(rgb_np, d=10, sigmaColor=75, sigmaSpace=75)
    denoised_img_np = np.dstack((denoised_rgb_np, alpha_np))
    denoised_img = Image.fromarray(denoised_img_np)
    return denoised_img

def calculate_contrast(image):
    grayscale = image.convert('L')
    stat = ImageStat.Stat(grayscale)
    contrast = stat.stddev[0]
    return contrast

def enhance_contrast(img, low_threshold=50, high_threshold=70, low_factor=2.0, high_factor=1.0):
    contrast = calculate_contrast(img)
    if contrast < low_threshold:
        factor = low_factor
    elif contrast > high_threshold:
        factor = high_factor
    else:
        factor = low_factor - (low_factor - high_factor) * (contrast - low_threshold) / (high_threshold - low_threshold)

    enhancer = ImageEnhance.Contrast(img)
    enhanced_img = enhancer.enhance(factor)
    return enhanced_img

def indexar_colores(image, n_clusters):
    # Convertir la imagen a modo RGB si no lo está
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convertir la imagen a una matriz numpy
    image_array = np.array(image)

    # Obtener la forma de la matriz de la imagen
    height, width, channels = image_array.shape

    # Redimensionar la matriz a una forma plana para que cada píxel se represente como un vector de características
    flattened_image_array = image_array.reshape((height * width, channels))

    # Inicializar el modelo KMeans con el número de clusters especificado
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)

    # Ajustar el modelo a los datos de la imagen
    kmeans.fit(flattened_image_array)

    # Obtener las etiquetas de los clústeres para cada píxel
    labels = kmeans.predict(flattened_image_array)

    # Asignar los colores de los clústeres a cada píxel en la imagen
    clustered_image = kmeans.cluster_centers_[labels]

    # Reformar la imagen para que tenga la misma forma que la imagen original
    clustered_image = clustered_image.reshape((height, width, channels))

    # Convertir la matriz numpy de vuelta a una imagen de PIL
    clustered_image = Image.fromarray(clustered_image.astype('uint8'), 'RGB')

    return clustered_image

def reduce_image(image, scale_factor=None, image_size=None):
    img = np.array(image)
    original_height, original_width = img.shape[:2]

    if image_size is not None:
        max_dimension = max(original_width, original_height)
        if original_width > original_height:
            new_width = image_size
            new_height = int(original_height * (image_size / original_width))
        else:
            new_height = image_size
            new_width = int(original_width * (image_size / original_height))
    elif scale_factor is not None:
        new_width = int(original_width * scale_factor)
        new_height = int(original_height * scale_factor)
    else:
        new_width, new_height = original_width, original_height

    # Usar cv2.INTER_NEAREST para reducir la imagen y preservar detalles
    small_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_NEAREST)
    small_image = Image.fromarray(small_img)
    return small_image

def enlarge_image(image, target_size):
    img = np.array(image)
    enlarged_img = cv2.resize(img, target_size, interpolation=cv2.INTER_NEAREST)
    enlarged_image = Image.fromarray(enlarged_img)
    return enlarged_image

def extract_palette(image):
    image = image.convert('RGB')
    data = np.array(image)
    data = data.reshape((-1, 3))
    unique_colors = np.unique(data, axis=0)
    n_colors = len(unique_colors)
    kmeans = KMeans(n_clusters=n_colors).fit(data)
    palette = kmeans.cluster_centers_.astype(int)
    return palette

def find_nearest_palette_color(color, palette):
    tree = KDTree(palette)
    _, idx = tree.query(color)
    return palette[idx]

def apply_palette(image, palette):
    image = image.convert('RGBA')
    data = np.array(image)
    rgb_data = data[..., :3]
    alpha_data = data[..., 3]
    new_rgb_data = rgb_data.reshape((-1, 3))
    
    for i in range(new_rgb_data.shape[0]):
        if alpha_data.reshape((-1,))[i] > 25.5:  # 10% de 255
            # Eliminar transparencia si la opacidad es mayor al 10%
            new_rgb_data[i] = find_nearest_palette_color(new_rgb_data[i], palette)
            alpha_data.reshape((-1,))[i] = 255
        elif alpha_data.reshape((-1,))[i] > 0:
            # Convertir en completamente transparente si la opacidad es menor o igual al 10%
            alpha_data.reshape((-1,))[i] = 0
    
    new_rgb_data = new_rgb_data.reshape(rgb_data.shape)
    new_data = np.dstack((new_rgb_data, alpha_data))
    return Image.fromarray(new_data.astype(np.uint8))

def list_palettes(palette_folder):
    palettes = os.listdir(palette_folder)
    palettes = [os.path.splitext(p)[0] for p in palettes if p.endswith('.png')]
    return palettes

def process_images(input_folder, output_folder, palette_image_path, palette_name, scale_factor, image_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if palette_image_path:
        palette_image = Image.open(palette_image_path)
        palette = extract_palette(palette_image)
    else:
        palette = None
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            output_filename = f"{palette_name}_{name}{ext}"
            output_path = os.path.join(output_folder, output_filename)
            original_image = Image.open(input_path)
            denoised_image = denoise_image(original_image)

            if palette_name!="original":
                contrast_image = enhance_contrast(denoised_image)
                small_image = reduce_image(contrast_image, scale_factor=scale_factor, image_size=image_size)
            else:
                small_image = reduce_image(denoised_image, scale_factor=scale_factor, image_size=image_size)
            
            if palette is not None:
                small_image_with_palette = apply_palette(small_image, palette)
            else:
                small_image_with_palette = small_image
            
            original_size = (original_image.width, original_image.height)
            pixel_art_image = enlarge_image(small_image_with_palette, original_size)
            
            if pixel_art_image.mode == 'RGBA':
                final_output_path = os.path.splitext(output_path)[0] + '.png'
                pixel_art_image.save(final_output_path, 'PNG')
            else:
                pixel_art_image.save(output_path, 'JPEG')
            
            print(f'Processed {input_path} and saved to {final_output_path}')

def process_one_image(filename, output_folder, palette_image_path, palette_name, scale_factor, image_size):
    input_folder=os.getcwd()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if palette_image_path:
        palette_image = Image.open(palette_image_path)
        palette = extract_palette(palette_image)
    else:
        palette = None
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(os.path.basename(filename))
        output_filename = f"{palette_name}_{name}{ext}"
        output_path = os.path.join(output_folder, output_filename)
        original_image = Image.open(input_path)
        denoised_image = denoise_image(original_image)
        
        if palette_name=="nes" or len(palette)<32:
            contrast_image = enhance_contrast(denoised_image)
            small_image = reduce_image(contrast_image, scale_factor=scale_factor, image_size=image_size)
        else:
            small_image = reduce_image(denoised_image, scale_factor=scale_factor, image_size=image_size)
       
            
        if palette is not None:
            small_image_with_palette = apply_palette(small_image, palette)
        else:
            small_image_with_palette = small_image
            
        original_size = (original_image.width, original_image.height)
        pixel_art_image = enlarge_image(small_image_with_palette, original_size)
            
        if pixel_art_image.mode == 'RGBA':
            final_output_path = os.path.splitext(output_path)[0] + '.png'
            pixel_art_image.save(final_output_path, 'PNG')
        else:
            pixel_art_image.save(output_path, 'JPEG')
            
        print(f'Processed {input_path} and saved to {final_output_path}')

def process_images_cluster(input_folder, output_folder, palette_image_path, palette_name, scale_factor, image_size, n_clusters):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if palette_image_path:
        palette_image = Image.open(palette_image_path)
        palette = extract_palette(palette_image)
    else:
        palette = None
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            name, ext = os.path.splitext(filename)
            output_filename = f"{palette_name}_{name}{ext}"
            output_path = os.path.join(output_folder, output_filename)
            original_image = Image.open(input_path)
            denoised_image = denoise_image(original_image)
            small_image = reduce_image(denoised_image, scale_factor=scale_factor, image_size=image_size)
            
            imagen_indexada = indexar_colores(small_image, n_clusters)
            
            original_size = (original_image.width, original_image.height)

            pixel_art_image = enlarge_image(imagen_indexada, original_size)

            final_output_path = os.path.splitext(output_path)[0] + '.png'

            if pixel_art_image.mode == 'RGBA':
                pixel_art_image.save(final_output_path, 'PNG')
            else:
                pixel_art_image.save(output_path, 'JPEG')
            
            print(f'Processed {input_path} and saved to {final_output_path}')

def process_one_image_cluster(filename, output_folder, palette_image_path, palette_name, scale_factor, image_size, n_clusters):
    input_folder = os.getcwd()
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    if palette_image_path:
        palette_image = Image.open(palette_image_path)
        palette = extract_palette(palette_image)
    else:
        palette = None
    
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        input_path = os.path.join(input_folder, filename)
        name, ext = os.path.splitext(os.path.basename(filename))
        output_filename = f"{palette_name}_{name}.png"
        output_path = os.path.join(output_folder, output_filename)
        original_image = Image.open(input_path)

        denoised_image = denoise_image(original_image)

        small_image = reduce_image(denoised_image, scale_factor=scale_factor, image_size=image_size)
        imagen_indexada = indexar_colores(small_image, n_clusters)

        original_size = (original_image.width, original_image.height)
        pixel_art_image = enlarge_image(imagen_indexada, original_size)

        # Guardar como PNG para evitar artefactos de compresión JPEG
        final_output_path = output_path
        pixel_art_image.save(final_output_path, 'PNG')
        print(f'Processed {input_path} and saved to {final_output_path}')

def create_output_directory(output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte imágenes a estilo pixel art aplicando una paleta de colores.")
    parser.add_argument("-p", "--palette_name", help="Nombre de la paleta a usar (sin extensión). Si no se proporciona, se usan los colores originales de la imagen.")
    parser.add_argument("-f", "--scale_factor", type=float, help="Factor de escala para la conversión a pixel art.")
    parser.add_argument("-s", "--image_size", type=int, help="Tamaño de la imagen para la conversión a pixel art (el valor más alto será utilizado conservando la relación de aspecto original).")
    parser.add_argument("-l", "--list_palettes", action="store_true", help="Lista todas las posibles paletas disponibles.")
    parser.add_argument("-d", "--input_folder", help="Nombre de directorio de entrada.")
    parser.add_argument("-o", "--output_folder", help="Nombre de directorio de salida.")
    parser.add_argument("-i", "--input_image", help="Nombre del archivo de imagen.")
    parser.add_argument("-n", "--clusters_number", help="Numero de colores a indexar con clusters.")

    args = parser.parse_args()

    if args.list_palettes:
        palette_folder = 'paletas'
        available_palettes = list_palettes(palette_folder)
        print("Lista de paletas disponibles:")
        for palette in available_palettes:
            print(palette)
        exit()


    if args.scale_factor is not None and args.image_size is not None:
        parser.error("No se pueden usar ambos argumentos -f (factor de escala) y -s (tamaño de la imagen) al mismo tiempo.")

    if args.palette_name is not None and args.clusters_number is not None:
        parser.error("No se pueden usar ambos argumentos -p (palette_name) y -n (numero de clusters) al mismo tiempo.")

    if args.input_folder is not None and args.input_image is not None:
        parser.error("No se pueden usar ambos argumentos -d (directorio de imagenes) y -i (nombre de la imagen) al mismo tiempo.")
    
    if args.input_folder is None and args.input_image is None:
        parser.error("Se tiene que utilizar al menos uno de los argumentos -d (directorio de imagenes) o -i (nombre de la imagen).")

    if args.scale_factor is None and args.image_size is None and args.palette_name is None and args.clusters_number is None:
        parser.error("Debe proporcionar al menos un argumento entre -f (factor de escala), -i (tamaño de la imagen) o -p (nombre de la paleta) o -n (numero de clusters).")

    output_folder = 'output_pixelart'
    palette_folder = 'paletas'

    if args.input_folder is not None and not os.path.isdir(args.input_folder):
        parser.error(f"El directorio de entrada '{args.input_folder}' no existe.")
    else:
        input_folder = args.input_folder

    if args.input_image is not None and not os.path.isfile(args.input_image):
        parser.error(f"El archivo de imagen de entrada '{args.input_image}' no existe.")
    else:
        input_image=args.input_image
    
    if args.output_folder:
        create_output_directory(args.output_folder)
        output_folder = args.output_folder

    if args.palette_name is not None and args.palette_name!="original" and args.clusters_number is None:
        palette_image_path = os.path.join(palette_folder, args.palette_name + '.png')
        if not os.path.exists(palette_image_path):
            print(f"Paleta {args.palette_name} no encontrada. Usando la paleta por defecto 'nes'.")
            palette_image_path = os.path.join(palette_folder, 'nes.png')
            palette_name = 'nes'
        else:
            palette_name = args.palette_name
    else:
        palette_image_path = None
        if args.clusters_number is None:
            palette_name = 'original'
        else:
            palette_name = 'clusters'
            clusters_number=int(args.clusters_number)

    if args.input_folder is not None and args.clusters_number is None:
         process_images(input_folder, output_folder, palette_image_path, palette_name, args.scale_factor, args.image_size)
         print("COMPLETADO")

    if args.input_image is not None and args.clusters_number is None:
        process_one_image(input_image, output_folder, palette_image_path, palette_name, args.scale_factor, args.image_size)
        print("COMPLETADO")

    if args.input_folder is not None and args.clusters_number is not None:
         process_images_cluster(input_folder, output_folder, palette_image_path, palette_name, args.scale_factor, args.image_size, clusters_number)
         print("COMPLETADO")

    if args.input_image is not None and args.clusters_number is not None:
        process_one_image_cluster(input_image, output_folder, palette_image_path, palette_name, args.scale_factor, args.image_size, clusters_number)
        print("COMPLETADO")


    
