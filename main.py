from cnnClassifier import logger
from cnnClassifier.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage_02_prepare_base_model import PrepareBaseModelTrainingPipeline
from cnnClassifier.pipeline.stage_03_model_trainer import ModelTrainingPipeline
from cnnClassifier.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline


STAGE_NAME="Data Ingestion Stage"

try:
    logger.info(f" ******** stage {STAGE_NAME} started")
    obj = DataIngestionTrainingPipeline()
    obj.main()
    logger.info(f" ********** stage {STAGE_NAME} completed")
except Exception as e:
    raise e

STAGE_NAME="Prepare Base Model Stage"

try:
    logger.info(f"******* stage {STAGE_NAME} started")
    obj = PrepareBaseModelTrainingPipeline()
    obj.main()
    logger.info(f"******** stage {STAGE_NAME} completed")
except Exception as e:
        raise e


STAGE_NAME=" Model Trainer Stage"

try:
    logger.info(f"******* stage {STAGE_NAME} started")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f"******** stage {STAGE_NAME} completed")
except Exception as e:
        raise e

STAGE_NAME=" Model Evaluation Stage"

try:
    logger.info(f"******* stage {STAGE_NAME} started")
    obj = ModelEvaluationPipeline()
    obj.main()
    logger.info(f"******** stage {STAGE_NAME} completed")
except Exception as e:
        raise e