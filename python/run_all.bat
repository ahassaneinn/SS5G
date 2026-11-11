@echo off
REM SS5G v3.0 -- whole Python pipeline (Windows). Open a prompt with your TensorFlow environment
REM active and run this file from the python folder. Put the two GAViST5G CSV files in ..\data first.
REM Every step can be re-run; finished tuning runs, models and seeds are skipped.
REM   set USE_RELEASED_TUNING=1 before running to skip the Bayesian optimisation and use the
REM   selected values in tuning\ (the values reported in the paper).
cd /d "%~dp0"
set DATA=..\data\CombinedALLGeo.csv
set GLOBAL=..\data\YouTubeCombinedGeo.csv
set SEEDS=0 1 2 3 4

echo ---- 1. prepare data
python eua_pipeline.py prepare --data %DATA% --work work || goto :fail
python eua_pipeline.py prepare --data %GLOBAL% --single-app YouTube --work work_global || goto :fail

echo ---- 2. smoke tests
python eua_pipeline.py smoke --work work --out results || goto :fail
python sota_models.py smoke --work work --out results || goto :fail

echo ---- 3. baselines
python eua_pipeline.py baselines --work work --out results || goto :fail
python eua_pipeline.py baselines --work work_global --out results_global || goto :fail

echo ---- 4. hyperparameter tuning (15 trials per model)
if "%USE_RELEASED_TUNING%"=="1" (
  if not exist results\tuning mkdir results\tuning
  copy /Y tuning\*.json results\tuning\ >nul
) else (
  python eua_pipeline.py tune --work work --out results --trials 15 || goto :fail
  python sota_models.py tune --work work --out results --trials 15 || goto :fail
)

echo ---- 5. training with the selected values: 11 models x 5 seeds
python eua_pipeline.py train --work work --out results --use-tuned --feature-sets full --seeds %SEEDS% || goto :fail
python sota_models.py train --work work --out results --use-tuned --feature-sets full --seeds %SEEDS% || goto :fail

echo ---- 6. inputs of the ns-3 evaluation (..\ns3\ns3_inputs)
python ..\ns3\make_ns3_inputs.py --work work --results results --out ..\ns3\ns3_inputs || goto :fail

echo ---- 7. architecture tables
python eua_pipeline.py describe --work work --out results --use-tuned || goto :fail
python sota_models.py describe --work work --out results --use-tuned || goto :fail

echo ---- 8. feature ablation, unseen users, international portion
python eua_pipeline.py train --work work --out results --use-tuned --models DNN GRU Transformer FEDformer --feature-sets no_rtt no_srv --seeds %SEEDS% || goto :fail
python eua_pipeline.py holdout --work work --out results --use-tuned --models GRU DNN || goto :fail
python eua_pipeline.py train --work work_global --out results_global --use-tuned --tuning-dir results\tuning --models all --feature-sets full --seeds 0 1 2 || goto :fail

echo ---- 9. inference latency: GPU, then CPU only
python eua_pipeline.py latency --work work --out results --use-tuned || goto :fail
python sota_models.py latency --work work --out results --use-tuned || goto :fail
set CUDA_VISIBLE_DEVICES=-1
python eua_pipeline.py latency --work work --out results --use-tuned || goto :fail
python sota_models.py latency --work work --out results --use-tuned || goto :fail
set CUDA_VISIBLE_DEVICES=

echo ---- 10. reports
python eua_pipeline.py report --out results || goto :fail
python eua_pipeline.py report --out results_global || goto :fail

echo DONE. Open results\REPORT.md; the ns-3 inputs are in ..\ns3\ns3_inputs
pause
exit /b 0
:fail
echo A step failed; see the messages above.
pause
exit /b 1
