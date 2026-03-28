import os
from textSummarizer.logging import logger


class DataValidation:
    def __init__(self, config):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        """Validates that all required dataset files/folders exist after ingestion."""
        try:
            validation_status = None
            all_files = os.listdir(
                os.path.join("artifacts", "data_ingestion", "samsum_dataset")
            )

            for required_file in self.config.ALL_REQUIRED_FILES:
                if required_file not in all_files:
                    validation_status = False
                    logger.info(
                        f"Validation failed: {required_file} is not found in the dataset"
                    )
                else:
                    validation_status = True
                    logger.info(f"Validation passed: {required_file} found")

            with open(self.config.STATUS_FILE, "w") as f:
                f.write(f"Validation status: {validation_status}")

            return validation_status

        except Exception as e:
            raise e
