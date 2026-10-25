@echo off

rem Parallel CUDA build using the Ninja generator.
rem NMake builds serially; Ninja uses all cores (2-5x faster).
rem Requires ninja on PATH: scoop install ninja  /  choco install ninja

where ninja >nul 2>&1
if errorlevel 1 (
    echo ninja not found on PATH. Install it first: scoop install ninja
    exit /b 1
)

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

rem rd /s /q build 2>nul
mkdir build 2>nul
cd build

cmake .. -G Ninja -DGGML_CUDA=ON
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%

cd ..