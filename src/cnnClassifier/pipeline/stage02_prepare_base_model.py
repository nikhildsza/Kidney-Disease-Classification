from cnnClassifier.config.configuration import ConfigurationManager
from cnnClassifier.components.prepare_base_model import PrepareBaseModel
from cnnClassifier import logger

# STAGE_NAME = "Prepare Base Model Stage"

class PrepareBaseModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        base_model_config = config.get_prepare_base_model_config()
        base_model = PrepareBaseModel(config=base_model_config)
        base_model.get_base_model()
        base_model.get_updated_model()

# if __name__=='__main__':
#     try:
#         logger.info(f'>>>>STAGE:{STAGE_NAME} has started<<<<<<<<')
#         obj = PrepareBaseModelTraingingPipeline()
#         obj.main()
#         logger.info(f'>>>>STAGE:{STAGE_NAME} has completed<<<<<<<<\n\n')
#     except Exception as e:
#         logger.exception(e)
#         raise e