# PHYS30046_Projects Instructions

# Setting up python environment #

0. ###  Assumes you have miniconda installed on MacOs Anaconda and Anaconda powershell installed on Windows 
    - To install miniconda on MacOs follow the instructions here https://docs.conda.io/projects/conda/en/stable/user-guide/install/macos.html making sure you download the Miniconda installer. 
    - To install anaconda and anaconda powershell on Windows follow the instructions here https://docs.conda.io/projects/conda/en/stable/user-guide/install/windows.html making sure you download the Anaconda installer. 
    - **Note any other installation of conda that suits you is also fine. The steps outlined below will make use of the command prompt (or shell) to startup a jupyter notebook session. If you are comfortable with any other method then please use that.** 
    
1. ###  You only need to do the below once 
    - MacOS: 
        - Open a terminal (eg by pressing "cmd"+"space" and typing "Terminal" or "iTerm". [You will use this terminal in subsequent steps when launching your jupyter notebook]
        - Assuming you have conda installed (if not follow the instructions at the top) type: 
            - `conda create -n hep-env python=3.11 uproot zfit matplotlib numpy scipy iminuit pandas jupyter` and hit enter
        - Now activate the environment by typing: 
            - `conda activate hep-env` and hit enter
        - Now install another package (or two) by typing:
            - `pip install mplhep`
     - Windows:
         - This assumes you have anaconda installed (if not follow the instructions at the top)
         - Press the windows key and search for "Anaconda PowerShell Prompt" and click on icon that comes up. You'll get a terminal with (base) at the beginning of the line. [You will use this Powershell prompt in subsequent steps when launching your jupyter notebook]
         - Now type:
             - `conda create -n hep-env python=3.11 uproot zfit matplotlib numpy scipy iminuit pandas jupyter` and hit enter
        - Now activate the environment by typing: 
            - `conda activate hep-env` and hit enter
        - Now install another package (or two) by typing:
            - `pip install mplhep`
           
 2. ### Launching jupyter notebook within correct conda environment. You do this every time you log on
     - MacOS:
         - From a terminal (MAor Anaconda type:
             - `conda activate hep-env` and press enter.
             - `jupyter notebook` and press enter. 
             - This should pop open a browser window. Navigate to the location of the .ipynb file and double click on it to open it. You can now edit and execute the python notebook on your browser
     - W
