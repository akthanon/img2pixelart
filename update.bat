@echo off
setlocal

REM Ruta del directorio donde se descargará y descomprimirá el repositorio
set REPO_DIR=%cd%\img2pixelart
echo %REPO_DIR%

REM URL del repositorio de GitHub
set GITHUB_REPO=https://github.com/calyseym/img2pixelart

REM URL del archivo zip de la última versión del repositorio
set ZIP_URL=%GITHUB_REPO%/archive/refs/heads/main.zip

REM Nombre del archivo zip descargado
set ZIP_FILE=img2pixelart.zip

REM Descarga el archivo zip usando curl
echo Descargando la última versión del repositorio...
curl -L -o %ZIP_FILE% %ZIP_URL%

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

REM Mueve los archivos a la carpeta deseada
move img2pixelart-main main_code

REM Elimina el archivo zip descargado
del %ZIP_FILE%

echo Actualización completada.
pause

endlocal
