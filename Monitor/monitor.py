#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#You have to import all of this, by adding the module intelpython3, and pip install everything. After that, u need the cuda module, or it can't work !

from threading import Thread
import time
import psutil
import pynvml

import eco2ai
from carbontracker.tracker import CarbonTracker
import os
import cpuinfo
import re
import subprocess
import sys
from datetime import datetime
import pandas as pd
import FR
import warnings
warnings.filterwarnings("ignore")


class Monitor(Thread):
    global total_cpu, count, cpu_now, total_gpu, gpu_now,  tracker_codecarbon, tracker_eco2ai, tracker_carbontracker, total_time, start_time, total_ram,num_gpus, pid, dossier, carbon, carboncount

    def __init__(self, delay):
        super(Monitor, self).__init__()
        #Création du dossier co2 pour ranger les données, un sous dossier ave la date sera fait ensuite.
        rep="co2"
        if (not(os.path.exists(rep))):
            os.mkdir(rep)
        #Variable pour continuer le while du run.
        self.stopped = False
        #delay pour faire les mesures, dans l'executable il est de 10s, si on augmente on risque de consommer plus, et si on diminue, on aura des mesures d'usage factor peu fiable (donc si on n'utilise pas GreenAlgorithm, on peut augmenter en théorie)
        self.delay = delay 
        self.total_cpu = 0
        self.count = 0 
        self.total_gpu = 0
        self.total_ram = 0
        self.carboncount = 0
        self.carbon = 0
        #sous dossier :
        self.dossier = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
        if (not(os.path.exists("co2/"+self.dossier))):
            os.mkdir("co2/"+self.dossier)
        #Lancement de CodeCarbon
        from codecarbon import EmissionsTracker
        self.tracker_codecarbon = EmissionsTracker(output_dir='co2/'+self.dossier, output_file='codecarbon_emissions.csv', tracking_mode = 'process')
        
        #Lancement de eco2ai
        self.tracker_eco2ai = eco2ai.Tracker(project_name="My_default_project_name", experiment_description="We trained...", file_name="co2/"+self.dossier+"/eco2ai_emissions.csv")
        
        #Lancement de CarbonTracker 
        #self.tracker_carbontracker = CarbonTracker(epochs=1, log_dir="co2/"+self.dossier, log_file_prefix="carbontracker_emissions")
        

        # Initialiser la bibliothèque pynvml
        try : 
            pynvml.nvmlInit()
            self.num_gpus = pynvml.nvmlDeviceGetCount()
        except pynvml.NVMLError_FunctionNotFound:
            # La fonction n'est pas trouvée, la carte graphique Nvidia n'est pas disponible
            self.num_gpus = 0
        except pynvml.NVMLError as e:
            # Une autre erreur liée à pynvml s'est produite
            self.num_gpus = 0
        self.start_time = time.time()
        self.tracker_codecarbon.start()
        
        self.tracker_eco2ai.start()
        #self.tracker_carbontracker.epoch_start()
        self.pid = os.getpid()
        
        self.start()

    def run(self):
        time.sleep(10)
        while not self.stopped:
            #Ici, on cherche l'usage factor du GPU. Donc on lance prend la mesure toutes les 10s et on prendra ensuite la moyenne.
            self.gpu_now = 0
            # Parcourir chaque GPU et trouver celui associé au processus en cours
            try :
                for i in range(self.num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    # Vérifier si le PID correspond
                    for info in process_info:
                        if info.pid == self.pid:
                            self.gpu_now = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            break
            except pynvml.NVMLError_FunctionNotFound:
            # La fonction n'est pas trouvée, la carte graphique Nvidia n'est pas disponible
                pass
            except pynvml.NVMLError as e:
                # Une autre erreur liée à pynvml s'est produite
                pass
            if (self.count%(3600/self.delay)==0):
                try :
                    self.carbon += self.carbonIntensity()
                except Exception :
                    self.carbon += self.get_value_for_data("France")
                self.carboncount += 1 
            self.total_gpu += self.gpu_now
            #Pareil, mais usage factor du CPU
            cpu_usage = psutil.cpu_percent(interval=None, percpu=False)
            self.total_ram += psutil.virtual_memory().used/(1024**3)
            self.total_cpu +=  cpu_usage
            self.count += 1
            time.sleep(self.delay)

    def stop(self):
        #On stoppe tous les trackers
        self.stopped = True
        self.tracker_eco2ai.stop()
        #self.tracker_carbontracker.epoch_end()
        #self.tracker_carbontracker.stop()
        self.tracker_codecarbon.stop()
        self.total_time = (time.time() - self.start_time)/60 #En minutes
        cpu_name =cpuinfo.get_cpu_info()['brand_raw']
        nb_cpu = self.count_cpus()
        gpu_name = None
        for i in range(self.num_gpus):
            gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
        average_ram = self.total_ram/self.count
        cpu_usage = self.total_cpu/self.count/100
        gpu_usage = self.total_gpu/self.count/100
        GA = self.GreenAlgo(self.total_time, cpu_name, nb_cpu, gpu_name, self.num_gpus, average_ram, cpu_usage, gpu_usage)
        
        fichier = open("co2/"+self.dossier+"/GreenAlgorithm_emissions.txt", "w")
        if GA is not None :
            fichier.write("This file is the most precise\n")
            fichier.write("Stats :\n{} g de co2.\n{} kWh".format(GA[0], GA[1]))
            self.fichier_total_GA(GA[0], GA[1])
        else :
            #On écrit les output à mettre à la main dans GreenAlgorithm ici. C'est la mesure la plus précise que l'on puisse avoir.
            fichier.write("If you want more precise and efficient stats, enter your data in the csv file.")
            fichier.write("Stats for GreenAlgorithm: http://calculator.green-algorithms.org/\n")
            fichier.write("Run time: {} minutes\n".format(self.total_time))
            fichier.write("Type of cores: Both\n")
            fichier.write("CPUs:\n")
            fichier.write("Number of cores: {}\n".format(nb_cpu))
            fichier.write("Model: {}\n".format(cpu_name))
            fichier.write("GPUs:\n")
            fichier.write("Number of GPUs: {}\n".format(self.num_gpus))
            for i in range(self.num_gpus):
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
                fichier.write("GPU {}: {}\n".format(i+1, gpu_name))
            fichier.write("Mémoire RAM moyenne utilisée: {} Go\n".format(average_ram))
            fichier.write("The CPU usage factor is: {}\n".format(cpu_usage))
            fichier.write("The GPU usage factor is: {}\n".format(gpu_usage))
            self.fichier_total()
        fichier.close()
        
        


      
#FROM CodeCarbon ! But I changed the cpu count because we need the physical one, not the logicals.
    def count_cpus(self) -> int:
        if os.environ.get("SLURM_JOB_ID") is None:
            return psutil.cpu_count(logical=False)

        try:
            scontrol = subprocess.check_output(
                ["scontrol show job $SLURM_JOBID"], shell=True
            ).decode()
        except subprocess.CalledProcessError:
            logger.warning(
                "Error running `scontrol show job $SLURM_JOBID` "
                + "to count SLURM-available cpus. Using the machine's cpu count."
            )
            return psutil.cpu_count(logical=False)

        num_cpus_matches = re.findall(r"NumCPUs=\d+", scontrol)

        if len(num_cpus_matches) == 0:
            logger.warning(
                "Could not find NumCPUs= after running `scontrol show job $SLURM_JOBID` "
                + "to count SLURM-available cpus. Using the machine's cpu count."
            )
            return psutil.cpu_count(logical=False)

        if len(num_cpus_matches) > 1:
            logger.warning(
                "Unexpected output after running `scontrol show job $SLURM_JOBID` "
                + "to count SLURM-available cpus. Using the machine's cpu count."
            )
            return psutil.cpu_count(logical=False)

        num_cpus = num_cpus_matches[0].replace("NumCPUs=", "")
        return int(num_cpus)
    
    def fichier_total(self):

        # Chemin vers le premier fichier CSV
        chemin_fichier_source = 'co2/'+self.dossier+'/codecarbon_emissions.csv'

        # Vérifier si le fichier "total.csv" existe déjà
        chemin_fichier_total = "co2/total.csv"
        if not os.path.exists(chemin_fichier_total):
            # Si le fichier n'existe pas, créer un DataFrame avec les colonnes "emissions" et "total_energy"
            df_total = pd.DataFrame(columns=["emissions (kg)", "total_energy (kWh)"])
            df_total.loc[0] = [0, 0]  # Ajouter une première ligne avec des valeurs initiales à zéro
        else:
            # Sinon, charger le contenu existant
            df_total = pd.read_csv(chemin_fichier_total)

        # Charger le premier fichier CSV
        df_source = pd.read_csv(chemin_fichier_source)

        # Obtenir la valeur de la colonne "emissions" du premier fichier
        valeur_emissions = df_source["emissions"].values[0]

        # Obtenir la valeur de la colonne "energy_consumed" du premier fichier
        valeur_energy_consumed = df_source["energy_consumed"].values[0]

        # Ajouter les valeurs aux valeurs existantes dans les colonnes "emissions" et "total_energy" du fichier "total.csv"
        df_total.loc[0, "emissions (kg)"] += valeur_emissions
        df_total.loc[0, "total_energy (kWh)"] += valeur_energy_consumed

        # Enregistrer le résultat dans le fichier "total.csv"
        df_total.to_csv(chemin_fichier_total, index=False)

    def fichier_total_GA(self, valeur_emissions, valeur_energy_consumed):

        # Vérifier si le fichier "total.csv" existe déjà
        chemin_fichier_total = "co2/total.csv"
        if not os.path.exists(chemin_fichier_total):
            # Si le fichier n'existe pas, créer un DataFrame avec les colonnes "emissions" et "total_energy"
            df_total = pd.DataFrame(columns=["emissions (kg)", "total_energy (kWh)"])
            df_total.loc[0] = [0, 0]  # Ajouter une première ligne avec des valeurs initiales à zéro
        else:
            # Sinon, charger le contenu existant
            df_total = pd.read_csv(chemin_fichier_total)

        # Ajouter les valeurs aux valeurs existantes dans les colonnes "emissions" et "total_energy" du fichier "total.csv"
        df_total.loc[0, "emissions (kg)"] += valeur_emissions/1000
        df_total.loc[0, "total_energy (kWh)"] += valeur_energy_consumed/1000

        # Enregistrer le résultat dans le fichier "total.csv"
        df_total.to_csv(chemin_fichier_total, index=False)

    def GreenAlgo(self, time_code, cpu_name , number_core_code, gpu_name, number_gpu_code, ram_moyenne, CPU_usage, GPU_usage) :
        runTime=time_code
        numberCPUs=number_core_code
        numberGPUs=number_gpu_code
        CPUpower = self.get_value_for_data(cpu_name)
        if CPUpower is not None :
            if gpu_name is not None :
                GPUpower = self.get_value_for_data(gpu_name)
            else :
                GPUpower = 0
            if GPUpower is not None :
                memory=ram_moyenne
                power_memory = self.get_value_for_data("memory_power")
                usageCPU=CPU_usage
                usageGPU=GPU_usage
                PSF=1
                PUE_used = 1.67
                powerNeeded_CPU = PUE_used * numberCPUs * CPUpower * usageCPU
                powerNeeded_GPU = PUE_used * numberGPUs * GPUpower * usageGPU
                """if (powerNeeded_GPU==0) :
                    print("GPU utilisation : 0%")"""
                powerNeeded_core = powerNeeded_CPU + powerNeeded_GPU
                powerNeeded_memory = PUE_used * (memory * power_memory)
                powerNeeded = powerNeeded_core + powerNeeded_memory
                energyNeeded = (runTime/60) * powerNeeded * PSF / 1000
                carbonIntensity = self.carbon/self.carboncount
                carbonEmissions = energyNeeded * carbonIntensity
                if (carbonIntensity==51.28):
                    print("La clé API ne fonctionne pas, valeur par défaut générée. Si vous voulez un résultat précis, générez une clée sur https://opendata.reseaux-energies.fr/, et définissez la pour RESEAUX_ENERGIES_TOKEN")
                return [carbonEmissions, energyNeeded]
        return None

    def get_value_for_data(self, target_data):
        df = pd.read_csv("Monitor/data_GA.csv")
        row = df[df['data'] == target_data]
        if not row.empty:
            return float((row['value'].values[0]).replace(',','.'))
        return None

    def changepid(self, pid):
        self.pid=pid

    def carbonIntensity(self):
        try :
            data_list = FR.fetch_production()
            entry = data_list[-1]
            total = 0
            total_production = sum(entry['production'].values())
            for prod in entry['production'] :
                total += entry['production'][prod]*self.get_value_for_data(prod)
            return total/total_production
        except KeyError :
            #print("La clé API ne fonctionne pas, valeur par défaut générée. Si vous voulez un résultat précis, générez une clée sur https://opendata.reseaux-energies.fr/, et définissez la pour RESEAUX_ENERGIES_TOKEN")
            return self.get_value_for_data("France")
        except Exception :
            #print("La clé API ne fonctionne pas, valeur par défaut générée. Si vous voulez un résultat précis, générez une clée sur https://opendata.reseaux-energies.fr/, et définissez la pour RESEAUX_ENERGIES_TOKEN")
            return self.get_value_for_data("France")






#Pour lancer la commande linux
def linux_command(command, monitor):
	
	result = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
	monitor.changepid(result.pid)
	for line in result.stdout:
		print(line.strip().decode())
	
	error_output = result.stderr.read()
	if error_output:
		print("Erreur lors de l'exécution de la commande:")
		print(error_output.decode())

	result.wait()

#En cas d'utilisation en executable :
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py 'votre_commande_linux'")
        sys.exit(1)
    monitor = Monitor(10)
    print("Start measuring...")
    linux_command(sys.argv[1], monitor)
    monitor.stop()
    print("Measurement completed, see the result in the co2 folder")

