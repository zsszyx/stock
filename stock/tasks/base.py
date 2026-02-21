from abc import ABC, abstractmethod
import logging
from stock.config import settings

# Setup standard logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

class BaseTask(ABC):
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(name)

    @abstractmethod
    def run(self, *args, **kwargs):
        """Main execution logic for the task."""
        pass

    def log_progress(self, message: str):
        self.logger.info(message)

    def log_error(self, message: str, error: Exception):
        self.logger.error(f"{message}: {str(error)}", exc_info=True)
