from cnnClassifier import logger
from cnnClassifier.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline
from cnnClassifier.pipeline.stage02_prepare_base_model import PrepareBaseModelTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f'>>>>STAGE:{STAGE_NAME} has started<<<<<<<<')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'>>>>STAGE:{STAGE_NAME} has completed<<<<<<<<\n\nx===================x')
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Prepare Base Model Stage"

try:
    logger.info(f'>>>>STAGE:{STAGE_NAME} has started<<<<<<<<')
    base_model = PrepareBaseModelTrainingPipeline()
    base_model.main()
    logger.info(f'>>>>STAGE:{STAGE_NAME} has completed<<<<<<<<\n\nx===================x')
except Exception as e:
    logger.exception(e)
    raise e