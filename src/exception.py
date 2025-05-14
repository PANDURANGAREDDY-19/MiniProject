import sys
import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc_tb = error_detail.exc_info()
    error_message = "Error occuredin program [{0}] at line [{1}] with error [{2}]".format(
        exc_tb.tb_frame.f_code.co_filename,
        exc_tb.tb_lineno,
        str(error)
    )
    return error_message
    
class CustomException(Exception):
    def __init__(self, error_message, error_detail:sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail=error_detail)

    def __str__(self):
        return self.error_message
    
if __name__ == "__main__":
    logging.info("Logging has started")