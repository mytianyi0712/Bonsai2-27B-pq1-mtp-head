@echo off
rem Build the mtp-dump tool inside the existing build-win tree (MSVC 18 + CUDA 13.3).
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set "PATH=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja;%PATH%"
cd /d D:\models\kvmem-llama.cpp
rem mtp-dump last: its exe stays locked while a dump is running, and ninja stops
rem scheduling after a failed link, so the other tools must come first.
cmake --build build-win --target draft-logits head-graft mtp-trace mtp-dump -j 8
if errorlevel 1 exit /b 1
echo === done ===
ls -la build-win\bin\mtp-dump.exe build-win\bin\draft-logits.exe build-win\bin\head-graft.exe build-win\bin\mtp-trace.exe
