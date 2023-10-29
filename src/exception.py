import sys

from src.logger import logging

class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys) -> None:
        super().__init__(error_message)
        self.error_message = self.error_message_builder(error_message,error_detail)
        logging.error(self.error_message)

    def __str__(self) -> str:
        return self.error_message
    
    def error_message_builder(self, error, error_detail:sys) -> str:
        """
        Custom error message builder
        """
        _,_,exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        error_message = f"Error occured in python script [{file_name}] line number [{exc_tb.tb_lineno}] error message [{str(error)}]"
        return error_message