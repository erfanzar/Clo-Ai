import os
import sys
import time
import argparse

pars = argparse.ArgumentParser()

pars.add_argument('-a', '--all', nargs='?', default=True, type=bool, action="store_true")
pars.add_argument('-p', '--pytorch', nargs='?', default=False, type=bool, action="store_true")
pars.add_argument('-m', '--matplotlib', nargs='?', default=False, type=bool, action="store_true")
pars.add_argument('-c', '--cv', nargs='?', default=False, type=bool, action="store_true")
pars.add_argument('-f', '--flask', nargs='?', default=False, type=bool, action="store_true")
pars.add_argument('-s', '--pandas', nargs='?', default=False, type=bool, action="store_true")
pars.add_argument('-u', '--upgrade', nargs='?', default=False, type=bool, action="store_true")

arg = pars.parse_args()

c = time.time()

UPGRADE_PIP = 'python -m pip install --upgrade pip'
INSTALL_PYTORCH = 'conda install pytorch torchvision torchaudio cpuonly -c pytorch'
INSTALL_PANDAS = 'pip install pandas'
INSTALL_CV = 'pip install opencv-python'
INSTALL_MATPLOTLIB = 'pip install matplotlib'
INSTALL_FLASK = 'pip install flask'

if __name__ == '__main__':
    if arg.upgrade is True or arg.all is True:
        sys.stdout.write('\r Upgrading pip requirements')
        os.system(UPGRADE_PIP)
        sys.stdout.flush()
        print('pip Upgraded')

    if arg.pytorch is True or arg.all is True:
        sys.stdout.write('\r installing pytorch requirements')
        os.system(INSTALL_PYTORCH)
        sys.stdout.flush()
        print('pytorch installed')

    if arg.pandas is True or arg.all is True:
        sys.stdout.write('\r installing Pandas requirements')
        os.system(INSTALL_PANDAS)
        sys.stdout.flush()
        print('pandas installed')

    if arg.cv is True or arg.all is True:
        sys.stdout.write('\r installing OpenCV requirements')
        os.system(INSTALL_CV)
        sys.stdout.flush()
        print('OpenCV installed')

    if arg.matplotlib is True or arg.all is True:
        sys.stdout.write('\r installing Matplotlib requirements')
        os.system(INSTALL_MATPLOTLIB)
        sys.stdout.flush()
        print('Matplotlib installed')

    if arg.flask is True or arg.all is True:
        sys.stdout.write('\r installing Flask requirements')
        os.system(INSTALL_FLASK)
        sys.stdout.flush()
        print('Flask installed')

    print(f'Done in {time.time() - c:.4f} secs')
