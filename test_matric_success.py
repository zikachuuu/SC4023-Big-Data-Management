import subprocess
import os

MAIN_PY_PATH = 'main.py'
PYTHON_EXEC = '/Users/imoeto/Desktop/School Coursework/SC4023 Project/venv/bin/python'
RESULTS_DIR = 'Results/'

os.makedirs(RESULTS_DIR, exist_ok=True)

for i in range(100):
    last_two = f'{i:02d}'
    matric_num = f'U22406{last_two}C'
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
        # It seems like there is a typo in your code snippet. The variable `te` is not defined or used
        # anywhere in the provided code. If you intended to ask about something else or if you have a
        # specific question related to the code, please provide more context or clarify your question.
        text=True,
        timeout=120
    )
    if proc.returncode != 0:
        print(f'  [ERROR] main.py failed for {matric_num}:\n{proc.stderr}')

print('All ScanResult files generated.')
