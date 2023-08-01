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
    # Installation de GPUtil avec pip
    pip install GPUtil

    # Vérification de l'installation
    python3 -c "import GPUtil ; print(GPUtil.__version__)"
fi

if python3 -c "import threading" &> /dev/null; then
    echo "threading est déjà installé."
else
    # Installation de threading avec pip
    pip install threading
fi
#python3 -c "import threading ; print(threading.__version__)"

if python3 -c "import time" &> /dev/null; then
    echo "time est déjà installé."
else
    # Installation de time avec pip
    pip install TIME-python
fi
#python3 -c "import time ; print(time.__version__)"

if python3 -c "import pynvml" &> /dev/null; then
    echo "pynvml est déjà installé."
else
    # Installation de pynvml avec pip
    pip install pynvml
fi
python3 -c "import pynvml ; print(pynvml.__version__)"

if python3 -c "import sklearn" &> /dev/null; then
    echo "sklearn est déjà installé."
else
    # Installation de sklearn avec pip
    pip install scikit-learn
fi
python3 -c "import sklearn ; print(sklearn.__version__)"

if python3 -c "import eco2ai" &> /dev/null; then
    echo "eco2ai est déjà installé."
else
    # Installation de eco2ai avec pip
    pip install eco2ai
fi
#python3 -c "import eco2ai ; print(eco2ai.__version__)"

if python3 -c "import carbontracker" &> /dev/null; then
    echo "carbontracker est déjà installé."
else
    # Installation de carbontracker avec pip
    pip install carbontracker
fi
#python3 -c "import carbontracker ; print(carbontracker.__version__)"

if python3 -c "import os" &> /dev/null; then
    echo "os est déjà installé."
else
    # Installation de os avec pip
    pip install os-sys
fi
#python3 -c "import os ; print(os.__version__)"

if python3 -c "import cpuinfo" &> /dev/null; then
    echo "cpuinfo est déjà installé."
else
    # Installation de cpuinfo avec pip
    pip install py-cpuinfo
fi
#python3 -c "import cpuinfo ; print(cpuinfo.__version__)"

if python3 -c "import re" &> /dev/null; then
    echo "re est déjà installé."
else
    # Installation de re avec pip
    pip install regex
fi
python3 -c "import re ; print(re.__version__)"

if python3 -c "import subprocess" &> /dev/null; then
    echo "subprocess est déjà installé."
else
    # Installation de re avec pip
    pip install subprocess
fi
#python3 -c "import subprocess ; print(subprocess.__version__)"

if python3 -c "import codecarbon" &> /dev/null; then
    echo "codecarbon est déjà installé."
else
    # Installation de re avec pip
    pip install codecarbon
fi
python3 -c "import codecarbon ; print(codecarbon.__version__)"