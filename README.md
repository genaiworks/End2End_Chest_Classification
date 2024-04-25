# End2End_Chest_Classification

<!-- Workflow -->
#Entity->Component->Pipeline
1. Update config.yaml
2. Update params.yaml
3. Update entity
4. Update configuration manager in src config
5. Update component
6. Update pipeline
7. Update main.py
9. Update dvc.yaml

MLFlow

DagsHub Credentials
MLFLOW_TRACKING_URI=https://dagshub.com/genaiworks/End2End_Chest_Classification.mlflow \
MLFLOW_TRACKING_USERNAME=genaiworks \
MLFLOW_TRACKING_PASSWORD=ec34733c1df0a2704cccff33c8adf2070636133f \
python script.py

os.environ["MLFLOW_TRACKING_URI"]="https://dagshub.com/genaiworks/End2End_Chest_Classification.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"]="genaiworks" 
os.environ["MLFLOW_TRACKING_PASSWORD"]="ec34733c1df0a2704cccff33c8adf2070636133f"

export MLFLOW_TRACKING_URI=https://dagshub.com/genaiworks/End2End_Chest_Classification.mlflow 
export MLFLOW_TRACKING_USERNAME=genaiworks 
export MLFLOW_TRACKING_PASSWORD=ec34733c1df0a2704cccff33c8adf2070636133f

Run Using
dvc repro 
dvc dag