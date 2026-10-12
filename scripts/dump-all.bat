@echo off
rem Full hidden-state dump for MTP head training (two passes, ~10M tokens).
setlocal
set "DUMP=D:\models\kvmem-llama.cpp\build-win\bin\mtp-dump.exe"
set "MODEL=D:\models\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"

echo ==== pass 1/2: agent sessions (8.5M tokens) ====
"%DUMP%" -m "%MODEL%" --list D:\models\mtp-corpus\filelist-sessions.txt --out D:\models\mtp-corpus\dump-sessions --chunk 2048 --ub 1024 --max-tokens 8500000 --ngl 99 --threads 8
if errorlevel 1 exit /b 1

echo ==== pass 2/2: zh supplement (1.5M tokens) ====
"%DUMP%" -m "%MODEL%" --list D:\models\mtp-corpus\filelist-supplement.txt --out D:\models\mtp-corpus\dump-supplement --chunk 2048 --ub 1024 --max-tokens 1500000 --ngl 99 --threads 8
if errorlevel 1 exit /b 1

echo ==== ALL DONE ====
