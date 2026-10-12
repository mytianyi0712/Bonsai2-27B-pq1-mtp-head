@echo off
rem Pass-2 self-distillation trace generation (remaining prompts).
rem Wrapped in a .bat because a login-shell environment resets PATH and the native
rem mtp-trace.exe then cannot resolve the Universal CRT DLLs.
rem   usage: run-pass2.bat [max-rows]
setlocal
set "ROOT=D:\models"
set "MAXROWS=%~1"
if "%MAXROWS%"=="" set MAXROWS=350000

"%ROOT%\kvmem-llama.cpp\build-win\bin\mtp-trace.exe" ^
    -m "%ROOT%\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf" ^
    --prompts "%ROOT%\mtp-corpus\trace-prompts-pass2.txt" ^
    --out "%ROOT%\mtp-corpus\traces-thinking-2" ^
    --template-file "%ROOT%\bonsai-chat-template.jinja" ^
    --max-new 700 --chunk 1024 --max-rows %MAXROWS% ^
    --ctk q4_0 --ctv q4_0 --ngl 99 --threads 8
exit /b %ERRORLEVEL%
