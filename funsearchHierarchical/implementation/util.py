import config as config_lib

# Logging function that prints to stdout and writes to a log file.
def LOG(message: str):
  print(message)
  log_file = config_lib.TEST_LOG_FILE if config_lib.test_env else config_lib.LOG_FILE
  with open(log_file, 'a') as f:
    f.write(f"{message}\n")
