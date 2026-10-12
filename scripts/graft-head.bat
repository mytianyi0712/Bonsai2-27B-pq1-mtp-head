@echo off
rem Graft a trained MTP head into a copy of the base model, then verify it.
rem
rem   graft-head.bat CKPT OUT_MODEL
rem     CKPT      trained checkpoint (D:\models\mtp-train\run1\head_final.pt)
rem     OUT_MODEL patched model path (e.g. D:\models\Ternary-Bonsai-2-27B-PTQ1_0-mtp-t1.gguf)
rem
rem Steps: export fp16 tensors -> quantize to each tensor's deployment type and patch
rem a byte-copy of the base model -> run the MTP draft graph and dump its logits.
setlocal
set "PY=D:\models\mtp-train\.venv\Scripts\python.exe"
set "GRAFT=D:\models\kvmem-llama.cpp\build-win\bin\head-graft.exe"
set "DRAFT=D:\models\kvmem-llama.cpp\build-win\bin\draft-logits.exe"
set "BASE=D:\models\Ternary-Bonsai-2-27B-PTQ1_0-mtp.gguf"
set "CKPT=%~1"
set "OUT=%~2"
if "%CKPT%"=="" ( echo usage: graft-head.bat CKPT OUT_MODEL & exit /b 1 )
if "%OUT%"==""   ( echo usage: graft-head.bat CKPT OUT_MODEL & exit /b 1 )

set "EXP=D:\models\mtp-train\head-export-last"
echo ==== 1/3 export %CKPT% -> %EXP%
"%PY%" D:\models\mtp-train\export_head.py --ckpt "%CKPT%" --out "%EXP%" || exit /b 1

echo ==== 2/3 graft -> %OUT%
"%GRAFT%" --src "%BASE%" --out "%OUT%" --weights "%EXP%" || exit /b 1

echo ==== 3/3 verify (MTP draft graph on the patched model)
rem 64 real token ids from the first dumped chunk
for /f "delims=" %%T in ('"%PY%" -c "import numpy as np;print(','.join(str(int(x)) for x in np.fromfile(r'D:\models\mtp-corpus\dump-sessions\chunks\tok_0000000.i32',dtype=np.int32)[:64]))"') do set "TOKS=%%T"
"%DRAFT%" -m "%OUT%" --tokens %TOKS% --out D:\models\mtp-train\graft-verify || exit /b 1
"%PY%" D:\models\mtp-train\verify_graft.py --val D:\models\mtp-train\graft-verify --ckpt "%CKPT%" || exit /b 1
echo ==== done: %OUT%
