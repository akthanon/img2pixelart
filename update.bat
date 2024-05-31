@echo off
setlocal

REM Verifica si curl está instalado
curl --version >nul 2>&1
set CURL_INSTALLED=%errorlevel%

REM Verifica si wget está instalado
wget --version >nul 2>&1
set WGET_INSTALLED=%errorlevel%

REM Verifica si se tiene curl o wget
if %CURL_INSTALLED% neq 0 if %WGET_INSTALLED% neq 0 (
    echo Error: No se encontró curl ni wget.
    echo Por favor, instale curl o wget y vuelva a intentarlo.
    exit /b 1
)

REM Ruta del directorio donde se descargará y descomprimirá el repositorio
set REPO_DIR=%cd%\img2pixelart
echo %REPO_DIR%

REM URL del repositorio de GitHub
set GITHUB_REPO=https://github.com/calyseym/img2pixelart

REM URL del archivo zip de la última versión del repositorio
set ZIP_URL=%GITHUB_REPO%/archive/refs/heads/main.zip

REM Nombre del archivo zip descargado
set ZIP_FILE=img2pixelart.zip

REM Descarga el archivo zip usando curl si está disponible, de lo contrario usa wget
if %CURL_INSTALLED% equ 0 (
    echo Descargando la última versión del repositorio con curl...
    curl -L -o %ZIP_FILE% %ZIP_URL%
) else (
    echo Descargando la última versión del repositorio con wget...
    wget -O %ZIP_FILE% %ZIP_URL%
)

REM Verifica si la descarga fue exitosa
if %errorlevel% neq 0 (
    echo Error al descargar el archivo.
    exit /b 1
)

REM Elimina el directorio anterior si existe
if exist %REPO_DIR% (
    echo Eliminando la versión anterior...
    rmdir /S /Q %REPO_DIR%
)

REM Descomprime el archivo zip descargado
echo Descomprimiendo el archivo...
tar -xf %ZIP_FILE% > nul

REM Elimina el archivo de actualizaicón para evitar errores, este es el único archivo que no se sobreescribe
del img2pixelart-main\update.bat

REM Mueve los archivos a la carpeta deseada
xcopy img2pixelart-main\* . /E /H /Y 
rd /s /q img2pixelart-main

REM Elimina el archivo zip descargado
del %ZIP_FILE%

echo Actualización completada.
pause

endlocal
