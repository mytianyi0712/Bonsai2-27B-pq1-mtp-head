@echo off
rem Train the MTP draft head on the dumped hidden-state corpus.
rem
rem   train-mtp.bat                 -> run1 (2 epochs over the full corpus)
rem   train-mtp.bat NAME EPOCHS     -> custom run directory / epochs
rem
rem Data: D:\models\mtp-corpus\dump-sessions (+ dump-supplement when present)
rem Out:  D:\models\mtp-train\<NAME>   (head_step*.pt, head_final.pt)
setlocal
set "PY=D:\models\mtp-train\.venv\Scripts\python.exe"
set "NAME=%~1"
set "EP=%~2"
if "%NAME%"=="" set NAME=run1
if "%EP%"=="" set EP=2

set "DATA=--data D:\models\mtp-corpus\dump-sessions"
if exist "D:\models\mtp-corpus\dump-supplement\manifest.jsonl" set "DATA=%DATA% --data D:\models\mtp-corpus\dump-supplement"

"%PY%" D:\models\mtp-train\train_mtp.py %DATA% ^
    --out D:\models\mtp-train\%NAME% ^
    --device cuda --seq 512 --batch 2 --epochs %EP% --lr 8e-5 --qat deploy --accum 8 --val-every 1000
