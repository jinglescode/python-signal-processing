import json
import os
from datetime import datetime
from pathlib import Path

class Logger():
    def __init__(
        self, 
        log_dir="run_logs",
        filename_postfix=None,
    ):
        # create dir if does not exist
        Path(log_dir).mkdir(parents=True, exist_ok=True)

        # get this log path
        now = datetime.now() 
        date_time = now.strftime("%Y_%m_%d-%H_%M_%S")
        
        filename = date_time+"-"+filename_postfix if filename_postfix is not None else date_time

        self.log_path = os.path.join(log_dir, filename+".txt")

    def write_to_log(self, content, break_line=False):
        
        content = str(content)
        
        with open(self.log_path, 'a') as log_file:
            tofile = content + "\n"
            if break_line:
                tofile = "\n" + tofile
                
            log_file.write(tofile)
