import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import subprocess
import os

def select_input_folder():
    folder_path = filedialog.askdirectory()
    entry_input_folder.delete(0, tk.END)
    entry_input_folder.insert(tk.END, folder_path)

def select_output_folder():
    folder_path = filedialog.askdirectory()
    entry_output_folder.delete(0, tk.END)
    entry_output_folder.insert(tk.END, folder_path)

def select_input_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.png;*.jpg;*.jpeg")])
    entry_input_image.delete(0, tk.END)
    entry_input_image.insert(tk.END, file_path)

def update_scale_factor(scl_factor):
    entry_scale_factor.delete(0, tk.END)
    entry_scale_factor.insert(tk.END, scl_factor)

def update_image_size(img_size):
    entry_image_size.delete(0, tk.END)
    entry_image_size.insert(tk.END, img_size)

def process_images():
    # Change button text to "Processing..."
    button_process.config(text="Processing...")
    root.update()

    # Rest of the code remains the same
    input_folder = entry_input_folder.get()
    output_folder = entry_output_folder.get()
    palette_name = selected_palette.get()
    scale_factor = entry_scale_factor.get()
    image_size = entry_image_size.get()
    input_image = entry_input_image.get()

    if not input_folder and not input_image:
        messagebox.showerror("Error", "Debe seleccionar una carpeta de entrada o una imagen de entrada.")
        return

    if input_folder and input_image:
        messagebox.showerror("Error", "Solo se puede seleccionar una carpeta de entrada o una imagen de entrada, no ambas.")
        return

    command = ["python", "img2pixelart.py"]

    if output_folder:
        command.extend(["-o", output_folder])

    if palette_name:
        command.extend(["-p", palette_name])

    if scale_factor_option.get() == 1:
        if scale_factor and float(scale_factor) != 0:
            command.extend(["-f", scale_factor])
    else:
        if image_size and float(image_size) != 0:
            command.extend(["-s", image_size])

    if input_folder:
        command.extend(["-d", input_folder])
    else:
        command.extend(["-i", input_image])

    try:
        subprocess.run(command, check=True)
        messagebox.showinfo("Proceso terminado", "El procesamiento de imágenes ha finalizado correctamente.")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("Error", f"Error al procesar las imágenes: {e}")

    # Change button text back to "Process Images" after processing is finished
    button_process.config(text="Process Images")
    root.update()


def clear_input_text():
    if input_choice.get() == 1:
        entry_input_image.delete(0, tk.END)
    else:
        entry_input_folder.delete(0, tk.END)

def toggle_input_option():
    clear_input_text()  # Clear input text when selection changes

    if input_choice.get() == 1:
        entry_input_folder.config(state="normal")
        button_input_folder.config(state="normal")
        entry_input_image.config(state="disabled")
        button_input_image.config(state="disabled")
    else:
        entry_input_folder.config(state="disabled")
        button_input_folder.config(state="disabled")
        entry_input_image.config(state="normal")
        button_input_image.config(state="normal")


def toggle_scale_factor_option():
    if scale_factor_option.get() == 1:
        entry_scale_factor.config(state="normal")
        scale_factor_slider.config(state="normal")
        entry_image_size.config(state="disabled")
        image_size_slider.config(state="disabled")
    else:
        entry_scale_factor.config(state="disabled")
        scale_factor_slider.config(state="disabled")
        entry_image_size.config(state="normal")
        image_size_slider.config(state="normal")

root = tk.Tk()
root.title("Pixel Art Converter")

# Input choice check buttons
input_choice = tk.IntVar()
input_choice.set(1)  # Default to Input Folder

input_folder_checkbutton = tk.Checkbutton(root, text="Input Folder", variable=input_choice, onvalue=1, offvalue=2, command=toggle_input_option)
input_folder_checkbutton.grid(row=0, column=0, sticky="w")

input_image_checkbutton = tk.Checkbutton(root, text="Input Image", variable=input_choice, onvalue=2, offvalue=1, command=toggle_input_option)
input_image_checkbutton.grid(row=1, column=0, sticky="w")

# Input folder
label_input_folder = tk.Label(root, text="Input Folder:")
label_input_folder.grid(row=0, column=1)
entry_input_folder = tk.Entry(root, width=50)
entry_input_folder.grid(row=0, column=2)
button_input_folder = tk.Button(root, text="Select", command=select_input_folder)
button_input_folder.grid(row=0, column=3)

# Input image
label_input_image = tk.Label(root, text="Input Image:")
label_input_image.grid(row=1, column=1)
entry_input_image = tk.Entry(root, width=50)
entry_input_image.grid(row=1, column=2)
button_input_image = tk.Button(root, text="Select", command=select_input_image)
button_input_image.grid(row=1, column=3)

# Output folder
label_output_folder = tk.Label(root, text="Output Folder:")
label_output_folder.grid(row=2, column=0, columnspan=2)
entry_output_folder = tk.Entry(root, width=50)
entry_output_folder.grid(row=2, column=2)
button_output_folder = tk.Button(root, text="Select", command=select_output_folder)
button_output_folder.grid(row=2, column=3)

# Palette name
label_palette_name = tk.Label(root, text="Palette Name:")
label_palette_name.grid(row=3, column=0, columnspan=2)

palette_folder = "paletas"
palettes = ["original"] + [p[:-4] for p in os.listdir(palette_folder) if p.endswith(".png")]
selected_palette = tk.StringVar(root)
selected_palette.set(palettes[0])  # Default value

palette_menu = tk.OptionMenu(root, selected_palette, *palettes)
palette_menu.grid(row=3, column=2)

# Scale factor choice check buttons
scale_factor_option = tk.IntVar()
scale_factor_option.set(0)  # Default to Image Size

scale_factor_checkbutton = tk.Checkbutton(root, text="Scale Factor", variable=scale_factor_option, onvalue=1, offvalue=0, command=toggle_scale_factor_option)
scale_factor_checkbutton.grid(row=4, column=0, sticky="w")

image_size_checkbutton = tk.Checkbutton(root, text="Image Size", variable=scale_factor_option, onvalue=0, offvalue=1, command=toggle_scale_factor_option)
image_size_checkbutton.grid(row=5, column=0, sticky="w")

# Scale factor
label_scale_factor = tk.Label(root, text="Scale Factor:")
label_scale_factor.grid(row=4, column=1)

entry_scale_factor = tk.Entry(root, width=10)
entry_scale_factor.grid(row=4, column=2)
entry_scale_factor.insert(tk.END, "0.2")

scale_factor_slider = tk.Scale(root, from_=0.1, to=1, orient="horizontal", resolution=0.1, length=200, command=update_scale_factor)
scale_factor_slider.grid(row=4, column=3)
scale_factor_slider.set(0.2)

# Image size
label_image_size = tk.Label(root, text="Image Size:")
label_image_size.grid(row=5, column=1)

entry_image_size = tk.Entry(root, width=10)
entry_image_size.grid(row=5, column=2)
entry_image_size.insert(tk.END, "128")

image_size_slider = tk.Scale(root, from_=16, to=800, orient="horizontal", resolution=16, length=200, command=update_image_size)
image_size_slider.grid(row=5, column=3)
image_size_slider.set(128)

# Process button
button_process = tk.Button(root, text="Process Images", command=process_images)
button_process.grid(row=6, column=2)

toggle_input_option()  # Ensure initial state is correct
toggle_scale_factor_option()  # Ensure initial state is correct

root.mainloop()
