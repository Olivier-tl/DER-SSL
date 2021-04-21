import os
import subprocess
import time

OUTPUT_PATH = 'output/wandb/'

while True:
    runs = os.listdir(OUTPUT_PATH)
    for run in runs:
        path = os.path.join(OUTPUT_PATH, run)
        if not os.path.isdir(path) or not run.startswith('offline'):
            continue
        try:
            subprocess.run(['wandb', 'sync', path])
        except Exception as e:
            print('error : ', e)

    time.sleep(600)
