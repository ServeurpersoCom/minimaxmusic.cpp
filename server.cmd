@echo off

set PATH=%~dp0build\Release;%PATH%

rem Multi-GPU: add --device <name> below (CUDA0, CUDA1, Vulkan0, CPU)

mm-server.exe ^
    --host 0.0.0.0 ^
    --port 8086 ^
    --models .\models ^
    --device CUDA0 ^
    --max-batch 1

pause
