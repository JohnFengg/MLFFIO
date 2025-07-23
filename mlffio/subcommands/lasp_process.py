import subprocess,sys,os

def run_evt(args=None):
    path=os.path.join(os.path.dirname(__file__),'lasp_train_evt')
    subprocess.run(['bash',path])

def run_loss(args=None):
    path=os.path.join(os.path.dirname(__file__),'lasp_train_loss')
    subprocess.run(['bash',path])

def run_plot(args=None):
    path=os.path.join(os.path.dirname(__file__),'lasp_train_plot.py')
    subprocess.run([sys.executable,path])

def run_md(args=None):
    path=os.path.join(os.path.dirname(__file__),'md_plot.py')
    subprocess.run([sys.executable,path])