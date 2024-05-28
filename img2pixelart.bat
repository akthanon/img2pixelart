@echo off
setlocal

rem Verificar si Python está instalado y obtener la ubicación del ejecutable
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo Python no está instalado o no se puede encontrar en la ruta del sistema. Favor de instalar Python y añadirlo a la variable de entorno PATH.
    goto :end
)
set PYTHON_EXECUTABLE=python

rem Verificar si pip está instalado
%PYTHON_EXECUTABLE% -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    rem Verificar si curl está instalado
    curl --version >nul 2>&1
    if %errorlevel% equ 0 (
        rem Utilizar curl si está instalado
        set DOWNLOADER=curl
    ) else (
        rem Verificar si wget está instalado
        wget --version >nul 2>&1
        if %errorlevel% equ 0 (
            rem Utilizar wget si está instalado
            set DOWNLOADER=wget
        ) else (
            echo Curl o Wget no están instalados. Favor de descargar Curl, Wget o pip manualmente y luego instalar pip.
            goto :end
        )
    )
    echo Pip no está instalado. Descargando e instalando pip...
    %DOWNLOADER% https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    %PYTHON_EXECUTABLE% get-pip.py
)

rem Verificar e instalar las bibliotecas necesarias
%PYTHON_EXECUTABLE% -c "import numpy" >nul 2>&1
if %errorlevel% neq 0 (
    echo numpy no está instalado. Instalando numpy...
    %PYTHON_EXECUTABLE% -m pip install numpy
)

%PYTHON_EXECUTABLE% -c "import cv2" >nul 2>&1
if %errorlevel% neq 0 (
    echo opencv-python-headless no está instalado. Instalando opencv-python-headless...
    %PYTHON_EXECUTABLE% -m pip install opencv-python-headless
)

%PYTHON_EXECUTABLE% -c "import PIL" >nul 2>&1
if %errorlevel% neq 0 (
    echo pillow no está instalado. Instalando pillow...
    %PYTHON_EXECUTABLE% -m pip install pillow
)

%PYTHON_EXECUTABLE% -c "import sklearn" >nul 2>&1
if %errorlevel% neq 0 (
    echo scikit-learn no está instalado. Instalando scikit-learn...
    %PYTHON_EXECUTABLE% -m pip install scikit-learn
)

rem Verificar si el programa principal está presente
if not exist gui_img2pixelart.py (
    echo El archivo gui_img2pixelart.py no está presente en el directorio actual. Por favor, asegúrese de que el archivo esté presente y vuelva a ejecutar este script.
    goto :end
)

rem Ejecutar el programa
%PYTHON_EXECUTABLE% gui_img2pixelart.py

:end