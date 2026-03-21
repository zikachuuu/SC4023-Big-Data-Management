import subprocess
import os
import sys

MAIN_PY_PATH = 'main.py'
PYTHON_EXEC = sys.executable
RESULTS_DIR = 'Results/'

os.makedirs(RESULTS_DIR, exist_ok=True)

for i in range(100):
    last_two = f'{i:02d}'
    matric_num = f'01234{last_two}C'
    print(f'Generating ScanResult for {matric_num}...')
    # Remove old ScanResult file if exists
    scanresult_path = os.path.join(RESULTS_DIR, f'ScanResult_{matric_num}.csv')
    if os.path.exists(scanresult_path):
        os.remove(scanresult_path)
    # Run main.py with the matric_num as input
    proc = subprocess.run(
        [PYTHON_EXEC, MAIN_PY_PATH],
        input=matric_num + '\n',
        capture_output=True,
        text=True,
        timeout=120
    )
    if proc.returncode != 0:
        print(f'  [ERROR] main.py failed for {matric_num}:\n{proc.stderr}')

print('All ScanResult files generated.')
