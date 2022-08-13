import argparse
import os
import sys
import time

os.system('pip install colorama')

try:
    from colorama import Fore
except ImportError:
    pass
pars = argparse.ArgumentParser()

pars.add_argument('-a', '--all', default=True, action="store_true")
pars.add_argument('-p', '--pytorch', default=False, action="store_true")
pars.add_argument('-m', '--matplotlib', default=False, action="store_true")
pars.add_argument('-c', '--cv', default=False, action="store_true")
pars.add_argument('-f', '--flask', default=False, action="store_true")
pars.add_argument('-s', '--pandas', default=False, action="store_true")
pars.add_argument('-u', '--upgrade', default=False, action="store_true")
pars.add_argument('-b', '--boto3', default=False, action="store_true")
pars.add_argument('--model', default=False, action='store_true')

arg = pars.parse_args()

c = time.time()

UPGRADE_PIP = 'python -m pip install --upgrade pip'
INSTALL_PYTORCH = 'conda install pytorch torchvision torchaudio cpuonly -c pytorch'
INSTALL_PANDAS = 'pip install pandas'
INSTALL_CV = 'pip install opencv-python'
INSTALL_MATPLOTLIB = 'pip install matplotlib'
INSTALL_FLASK = 'pip install flask'

if __name__ == '__main__':

    MODEL_URL = 'https://drive.google.com/u/0/uc?id=1GpMyaiyvMBTbW5zr_L_cPW_61dzeHkg9&export=download&confirm=t&uuid=b8b79ab4-c8be-4370-bb31-15d363a7071b'

    if arg.model:

        sys.stdout.write(f'{Fore.LIGHTBLUE_EX} Downloading Ai Model')
        tm = time.time()
        if not os.path.exists('Detect-Models'):
            os.mkdir("Detect-Models")
        os.chdir('Detect-Models')
        os.system(f'curl {MODEL_URL}')
        sys.stdout.write(f'{Fore.YELLOW} Downloading Done in {time.time() - tm :.4} sec')
        
    if arg.upgrade is True or arg.all is True:
        sys.stdout.write(f'\r {Fore.CYAN} Upgrading pip requirements')
        os.system(UPGRADE_PIP)
        sys.stdout.flush()
        print('pip Upgraded')

    if arg.pytorch is True or arg.all is True:
        sys.stdout.write(f'\r {Fore.CYAN} installing pytorch requirements')
        os.system(INSTALL_PYTORCH)
        sys.stdout.flush()
        print('pytorch installed')

    if arg.pandas is True or arg.all is True:
        sys.stdout.write(f'\r {Fore.CYAN} installing Pandas requirements')
        os.system(INSTALL_PANDAS)
        sys.stdout.flush()
        print('pandas installed')

    if arg.cv is True or arg.all is True:
        sys.stdout.write(f'\r {Fore.CYAN} installing OpenCV requirements')
        os.system(INSTALL_CV)
        sys.stdout.flush()
        print('OpenCV installed')

    if arg.matplotlib is True or arg.all is True:
        sys.stdout.write(f'\r {Fore.CYAN} installing Matplotlib requirements')
        os.system(INSTALL_MATPLOTLIB)
        sys.stdout.flush()
        print('Matplotlib installed')

    if arg.flask is True or arg.all is True:
        sys.stdout.write(f'\r {Fore.CYAN} installing Flask requirements')
        os.system(INSTALL_FLASK)
        sys.stdout.flush()
        print('Flask installed')
    if arg.boto3 or arg.all is True:
        sys.stdout.write(f'\r {Fore.CYAN} installing boto3 requirements')
        os.system('pip install boto3')
        sys.stdout.flush()
        print('boto3 installed')

    print(f'{Fore.RED} Done in {time.time() - c:.4f} secs')
