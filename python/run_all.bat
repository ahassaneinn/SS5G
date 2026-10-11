@echo off
REM ==========================================================================
REM  Full Python pipeline (Windows). Run from a prompt with your TensorFlow
REM  environment activated, inside this python folder:
REM       conda activate <your-tf-env>
REM       run_all.bat
REM  Put the two GAViST5G CSV files in ..\data first (see ..\data\README.md).
REM  Every stage writes to results\ (results_global\ for the international
REM  portion) and can be re-run; finished model/seed combinations are skipped.
REM ==========================================================================
setlocal
set DATA=..\data\dataset1.csv
set GLOBAL=..\data\dataset2.csv
set SEEDS=0 1 2 3 4

echo ---- 1. prepare data (about 30 s)
python eua_pipeline.py prepare --data "%DATA%" --work work || goto :err
python eua_pipeline.py prepare --data "%GLOBAL%" --single-app YouTube --work work_global || goto :err

echo ---- 2. smoke test: every model, 1 epoch, 6 users (a few minutes)
python eua_pipeline.py smoke --work work --out results || goto :err

echo ---- 3. baselines (seconds)
python eua_pipeline.py baselines --work work --out results || goto :err
python eua_pipeline.py baselines --work work_global --out results_global || goto :err

REM Optional: equal-budget Bayesian optimisation (several hours). Then add
REM --use-tuned to the train and latency commands below.
REM python eua_pipeline.py tune --work work --out results --models all --trials 15

echo ---- 4. seven DL models x 5 seeds
python eua_pipeline.py train --work work --out results --models all --feature-sets full --seeds %SEEDS% || goto :err

echo ---- 5. feature ablation: no RTT, no server history
python eua_pipeline.py train --work work --out results --models DNN GRU Transformer FEDformer --feature-sets no_rtt no_srv --seeds %SEEDS% || goto :err

echo ---- 6. unseen users: 5 user-disjoint folds
python eua_pipeline.py holdout --work work --out results --models GRU DNN || goto :err

echo ---- 7. international portion
python eua_pipeline.py train --work work_global --out results_global --models all --feature-sets full --seeds 0 1 2 || goto :err

echo ---- 8. inference latency: GPU, then CPU only
python eua_pipeline.py latency --work work --out results || goto :err
set CUDA_VISIBLE_DEVICES=-1
python eua_pipeline.py latency --work work --out results || goto :err
set CUDA_VISIBLE_DEVICES=

echo ---- 9. reports
python eua_pipeline.py report --out results
python eua_pipeline.py report --out results_global

echo ---- 10. inputs of the ns-3 evaluation (..\ns3\ns3_inputs)
python ..\ns3\make_ns3_inputs.py --python-dir . --out ..\ns3\ns3_inputs || goto :err

echo DONE. Open results\REPORT.md
goto :eof

:err
echo.
echo *** A step failed; see the error above. ***
exit /b 1
