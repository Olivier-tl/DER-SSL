import subprocess

for x in [0.5, 1,2,3,5,7]:
    subprocess.run(['python', 'main_sweep.py', '--beta=1', '--alpha=1', '--use_ssl=1', f'--ssl-alpha={x}', '--use_drl=0', '--model_name=resnet'])
for a in [1,2]:
    for l in [0.01, 0.001, 0.0001]:
        subprocess.run(['python', 'main_sweep.py', '--beta=1', '--alpha=1', '--use_ssl=0', '--use_drl=1', f'--drl_lambda={l}', f'--drl_alpha={a}', '--model_name=resnet'])