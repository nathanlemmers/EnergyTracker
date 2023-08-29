#!/bin/bash

# Vérification de l'installation existante de TensorFlow



if [ "$1" = "1" ]; then
module load userspace/all
module load intelpython3/3.7
fi


export RESEAUX_ENERGIES_TOKEN=e71b170247e3ff850045d94ab2647cecf6e62007ef3f367989b5e5f2


if python3 -c "import GPUtil" &> /dev/null; then
    echo "GPUtil est déjà installé."
    python3 -c "import GPUtil ; print(GPUtil.__version__)"
else
    # Installation de GPUtil avec python3 -m pip
    python3 -m pip install --user GPUtil

    # Vérification de l'installation
    python3 -c "import GPUtil ; print(GPUtil.__version__)"
fi

if python3 -c "import threading" &> /dev/null; then
    echo "threading est déjà installé."
else
    # Installation de threading avec python3 -m pip
    python3 -m pip install --user threading
fi
#python3 -c "import threading ; print(threading.__version__)"

if python3 -c "import time" &> /dev/null; then
    echo "time est déjà installé."
else
    # Installation de time avec python3 -m pip
    python3 -m pip install --user TIME-python
fi
#python3 -c "import time ; print(time.__version__)"

if python3 -c "import pynvml" &> /dev/null; then
    echo "pynvml est déjà installé."
else
    # Installation de pynvml avec python3 -m pip
    python3 -m pip install --user pynvml
fi
python3 -c "import pynvml ; print(pynvml.__version__)"

if python3 -c "import sklearn" &> /dev/null; then
    echo "sklearn est déjà installé."
else
    # Installation de sklearn avec python3 -m pip
    python3 -m pip install --user scikit-learn
fi
python3 -c "import sklearn ; print(sklearn.__version__)"

if python3 -c "import eco2ai" &> /dev/null; then
    echo "eco2ai est déjà installé."
else
    # Installation de eco2ai avec python3 -m pip
    python3 -m pip install --user eco2ai
fi
#python3 -c "import eco2ai ; print(eco2ai.__version__)"

if python3 -c "import carbontracker" &> /dev/null; then
    echo "carbontracker est déjà installé."
else
    # Installation de carbontracker avec python3 -m pip
    python3 -m pip install --user carbontracker
fi
#python3 -c "import carbontracker ; print(carbontracker.__version__)"

if python3 -c "import os" &> /dev/null; then
    echo "os est déjà installé."
else
    # Installation de os avec python3 -m pip
    python3 -m pip install --user os-sys
fi
#python3 -c "import os ; print(os.__version__)"

if python3 -c "import cpuinfo" &> /dev/null; then
    echo "cpuinfo est déjà installé."
else
    # Installation de cpuinfo avec python3 -m pip
    python3 -m pip install --user py-cpuinfo
fi
#python3 -c "import cpuinfo ; print(cpuinfo.__version__)"

if python3 -c "import re" &> /dev/null; then
    echo "re est déjà installé."
else
    # Installation de re avec python3 -m pip
    python3 -m pip install --user regex
fi
python3 -c "import re ; print(re.__version__)"

if python3 -c "import subprocess" &> /dev/null; then
    echo "subprocess est déjà installé."
else
    # Installation de re avec python3 -m pip
    python3 -m pip install --user subprocess
fi
#python3 -c "import subprocess ; print(subprocess.__version__)"

if python3 -c "import codecarbon" &> /dev/null; then
    echo "codecarbon est déjà installé."
else
    # Installation de re avec python3 -m pip
    python3 -m pip install --user codecarbon
fi
python3 -c "import codecarbon ; print(codecarbon.__version__)"