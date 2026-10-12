@echo off
rem Acceptance-rate A/B for the MTP draft head: shipped head vs trained head.
rem Both runs use greedy decoding, so the generated text must be IDENTICAL --
rem a diff means the draft changed the output distribution (a bug), while the
rem accept_pct lines are the actual comparison.
rem
rem usage: ab-accept.bat TRAINED_MODEL PROMPT_FILE [n_predict] [n_max]
setlocal
set "CLI=D:\models\kvmem-llama.cpp\build-win\bin\llama-kvmem-cli.exe"
set "BASE=D:\models\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
set "NEW=%~1"
set "PROMPT=%~2"
set "NP=%~3"
set "NMAX=%~4"
if "%NEW%"==""  ( echo usage: ab-accept.bat TRAINED_MODEL PROMPT_FILE [n_predict] [n_max] & exit /b 1 )
if "%PROMPT%"=="" ( echo usage: ab-accept.bat TRAINED_MODEL PROMPT_FILE [n_predict] [n_max] & exit /b 1 )
if "%NP%"==""   set NP=256
if "%NMAX%"=="" set NMAX=1
set KVMEM_TRACE=1

echo ==== A: shipped head (%BASE%) ====
"%CLI%" -m "%BASE%" -f "%PROMPT%" -n %NP% --spec-type draft-mtp --spec-draft-n-max %NMAX% --no-prompt > "%TEMP%\ab_a.txt" 2> "%TEMP%\ab_a.log"
type "%TEMP%\ab_a.log" | findstr /C:"spec_stats"
echo ==== B: trained head (%NEW%) ====
"%CLI%" -m "%NEW%" -f "%PROMPT%" -n %NP% --spec-type draft-mtp --spec-draft-n-max %NMAX% --no-prompt > "%TEMP%\ab_b.txt" 2> "%TEMP%\ab_b.log"
type "%TEMP%\ab_b.log" | findstr /C:"spec_stats"
echo ==== output equality ====
fc /b "%TEMP%\ab_a.txt" "%TEMP%\ab_b.txt" >nul && echo IDENTICAL generated text || echo DIFFERENT - investigate
