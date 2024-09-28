from cnnClassifier import logger
from cnnClassifier.pipeline.stage01_data_ingestion import DataIngestionTrainingPipeline

STAGE_NAME = "Data Ingestion Stage"

try:
    logger.info(f'>>>>STAGE:{STAGE_NAME} has started<<<<<<<<')
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.main()
    logger.info(f'>>>>STAGE:{STAGE_NAME} has completed<<<<<<<<\n\n')
except Exception as e:
    logger.exception(e)
    raise e
