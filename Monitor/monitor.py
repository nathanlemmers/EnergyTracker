#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#You have to import all of this, by adding the module intelpython3, and pip install everything. After that, u need the cuda module, or it can't work !

from threading import Thread
import time
import psutil
import pynvml

import eco2ai
#from carbontracker.tracker import CarbonTracker
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
    global total_cpu, count, cpu_now, total_gpu, gpu_now,  tracker_codecarbon, tracker_eco2ai, tracker_carbontracker, total_time, start_time, total_ram,num_gpus, pid, dossier, carbon, carboncount, new_time, pue, GPU_power_info, total_gpu_pynvml

    def __init__(self, delay, pue):
        
        self.pue = pue
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
        self.total_gpu_pynvml = 0
        self.GPU_power_info = 0
        self.total_ram = 0
        self.carboncount = 0
        self.carbon = 0
        self.new_time = datetime.now()
        #sous dossier :
        self.dossier = self.new_time.strftime("%Y.%m.%d_%H:%M:%S")
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
        current_user = psutil.Process().username()
        while not self.stopped:
            #Ici, on cherche l'usage factor du GPU. Donc on lance prend la mesure toutes les 10s et on prendra ensuite la moyenne.
            self.gpu_now_pynvml = 0
            gpu_now = 0
            gpu_mwatt = 0
            # Parcourir chaque GPU et trouver celui associé au processus en cours
            try :
                for i in range(self.num_gpus):
                    handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                    process_info = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    # Vérifier si le PID correspond
                    for info in process_info:
                        try :
                            user_name = psutil.Process(info.pid).username()
                        except psutil.NoSuchProcess:
                            user_name = None
                            # Si le processus n'existe plus, il peut être ignoré
                        if user_name == current_user:
                            self.gpu_now_pynvml = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
                            gpu_mwatt = pynvml.nvmlDeviceGetPowerUsage(handle)
                            break
            except pynvml.NVMLError_FunctionNotFound:
            # La fonction n'est pas trouvée, la carte graphique Nvidia n'est pas disponible
                pass
            except pynvml.NVMLError as e:
                # Une autre erreur liée à pynvml s'est produite
                pass
            gpu_now = self.nvidia_smi(current_user)
            self.GPU_power_info += gpu_mwatt
            self.total_gpu_pynvml += self.gpu_now_pynvml
            self.total_gpu += gpu_now
            #Pareil, mais usage factor du CPU
            user_processes = [p.info for p in psutil.process_iter(['pid', 'username', 'cpu_percent']) if p.info['username'] == current_user]
            cpu_total = sum(p['cpu_percent'] for p in user_processes)
            cpu_cores = psutil.cpu_count(logical=True)
            cpu_usage = cpu_total / cpu_cores if cpu_cores > 0 else 0
            self.total_cpu += cpu_usage
            user_processes = [p for p in psutil.process_iter(['pid', 'username']) if p.info['username'] == current_user]
            self.total_ram += sum(p.memory_info().rss / (1024 ** 3) for p in user_processes)
            self.count += 1
            if (self.count%(23.5*3600/self.delay)==0):
                try :
                    self.new_time = datetime.now()
                    self.carbonIntensity_day(self.new_time)
                except Exception :
                    self.carbon += self.get_value_for_data("France")*48
                    self.carboncount += 48 
            time.sleep(self.delay)
        

    def stop(self):
        current_user = psutil.Process().username()
        #On stoppe tous les trackers
        self.stopped = True
        self.carbonIntensity_date(self.new_time)
        self.total_time = (time.time() - self.start_time)/60
        self.tracker_eco2ai.stop()
        #self.tracker_carbontracker.epoch_end()
        #self.tracker_carbontracker.stop()
        self.tracker_codecarbon.stop()
        #En minutes
        cpu_name =cpuinfo.get_cpu_info()['brand_raw']
        nb_cpu = self.count_cpus()
        gpu_name = "No NVIDIA graphuc card found"
        try :
            for i in range(self.num_gpus):
                gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                gpu_name = pynvml.nvmlDeviceGetName(gpu_handle)
                if isinstance(gpu_name, bytes):
                    gpu_name = gpu_name.decode('utf-8')
        except pynvml.NVMLError_FunctionNotFound:
            # La fonction n'est pas trouvée, la carte graphique Nvidia n'est pas disponible
                pass
        except pynvml.NVMLError as e:
                # Une autre erreur liée à pynvml s'est produite
                pass 
        if (self.count!=0):     
            average_ram = self.total_ram/self.count
            cpu_usage = self.total_cpu/self.count/100
            gpu_usage_pynvml = self.total_gpu_pynvml/self.count/100
            gpu_usage = self.total_gpu/self.count/100
            gpu_mwatt = self.GPU_power_info/self.count
            if gpu_usage_pynvml ==0 :
                ratio_gpu = 0
            else : 
                ratio_gpu = gpu_usage/gpu_usage_pynvml
            GA = self.GreenAlgo(self.total_time, cpu_name, nb_cpu, self.num_gpus, average_ram, cpu_usage, gpu_mwatt, ratio_gpu)
            
            fichier = open("co2/"+self.dossier+"/GreenAlgorithm_emissions.txt", "w")
            if GA is not None :
                fichier.write("This file is the most precise\n")
                fichier.write("Stats :\n{} g de co2.\n{} kWh\n".format(GA[0], GA[1]))
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
                    if isinstance(gpu_name, bytes):
                        gpu_name = gpu_name.decode('utf-8')
                    fichier.write("GPU {}: {}\n".format(i+1, gpu_name))
                fichier.write("Mémoire RAM moyenne utilisée: {} Go\n".format(average_ram))
                fichier.write("The CPU usage factor is: {}\n".format(cpu_usage))
                fichier.write("The GPU usage factor is: {}\n".format(gpu_usage))
                self.fichier_total()
            fichier.close()
            return True
        else :
            return False
        
        


      
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
        chemin_fichier_source = 'co2/'+self.dossier+'/eco2ai_emissions.csv'

        # Vérifier si le fichier "total.csv" existe déjà
        chemin_fichier_total = "co2/total.csv"
        if not os.path.exists(chemin_fichier_total):
            # Si le fichier n'existe pas, créer un DataFrame avec les colonnes "emissions" et "total_energy"
            df_total = pd.DataFrame(columns=["total emissions (kg)", "total energy (kWh)", "last_run emissions (kg)", "last_run energy (kWh)"])
            df_total.loc[0] = [0, 0, 0, 0]  # Ajouter une première ligne avec des valeurs initiales à zéro
        else:
            # Sinon, charger le contenu existant
            df_total = pd.read_csv(chemin_fichier_total)

        # Charger le premier fichier CSV
        df_source = pd.read_csv(chemin_fichier_source)

        # Obtenir la valeur de la colonne "emissions" du premier fichier
        valeur_emissions = df_source["CO2_emissions(kg)"].values[0]

        # Obtenir la valeur de la colonne "energy_consumed" du premier fichier
        valeur_energy_consumed = df_source["power_consumption(kWh)"].values[0]

        # Ajouter les valeurs aux valeurs existantes dans les colonnes "emissions" et "total_energy" du fichier "total.csv"
        df_total.loc[0, "total emissions (kg)"] += valeur_emissions
        df_total.loc[0, "total energy (kWh)"] += valeur_energy_consumed
        df_total.loc[0, "last_run emissions (kg)"] = valeur_emissions
        df_total.loc[0, "last_run energy (kWh)"] = valeur_energy_consumed

        # Enregistrer le résultat dans le fichier "total.csv"
        df_total.to_csv(chemin_fichier_total, index=False)

    def fichier_total_GA(self, valeur_emissions, valeur_energy_consumed):

        # Vérifier si le fichier "total.csv" existe déjà
        chemin_fichier_total = "co2/total.csv"
        if not os.path.exists(chemin_fichier_total):
            # Si le fichier n'existe pas, créer un DataFrame avec les colonnes "emissions" et "total_energy"
            df_total = pd.DataFrame(columns=["total emissions (kg)", "total energy (kWh)", "last_run emissions (kg)", "last_run energy (kWh)"])
            df_total.loc[0] = [0, 0, 0, 0]  # Ajouter une première ligne avec des valeurs initiales à zéro
        else:
            # Sinon, charger le contenu existant
            df_total = pd.read_csv(chemin_fichier_total)

        # Ajouter les valeurs aux valeurs existantes dans les colonnes "emissions" et "total_energy" du fichier "total.csv"
        df_total.loc[0, "total emissions (kg)"] += valeur_emissions/1000
        df_total.loc[0, "total energy (kWh)"] += valeur_energy_consumed/1000
        df_total.loc[0, "last_run emissions (kg)"] = valeur_emissions/1000
        df_total.loc[0, "last_run energy (kWh)"] = valeur_energy_consumed/1000

        # Enregistrer le résultat dans le fichier "total.csv"
        df_total.to_csv(chemin_fichier_total, index=False)

    def GreenAlgo(self, time_code, cpu_name , number_core_code, number_gpu_code, ram_moyenne, CPU_usage, GPU_mwatt, ratio_GPU) :
        runTime=time_code
        numberCPUs=number_core_code
        numberGPUs=number_gpu_code
        CPUpower = self.get_value_for_data(cpu_name)
        if CPUpower is not None :
            memory=ram_moyenne
            power_memory = self.get_value_for_data("memory_power")
            usageCPU=CPU_usage
            PSF=1
            PUE_used = self.pue
            powerNeeded_CPU = PUE_used * numberCPUs * CPUpower * usageCPU
            powerNeeded_GPU = PUE_used * GPU_mwatt * ratio_GPU / 1000
            """if (powerNeeded_GPU==0) :
                print("GPU utilisation : 0%")"""
            powerNeeded_core = powerNeeded_CPU + powerNeeded_GPU
            powerNeeded_memory = PUE_used * (memory * power_memory)
            powerNeeded = powerNeeded_core + powerNeeded_memory
            energyNeeded = (runTime/60) * powerNeeded * PSF / 1000
            if (self.carboncount == 0) :
                carbonIntensity = 51.28
            else :
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

    def carbonIntensity_day(self, dateti):
        try :
            if (dateti.minute<30) :
                 date = dateti.replace(minute=0)
            else :
                 date = dateti.replace(minute=30)
            data_list = FR.fetch_production(target_datetime=date)
            date = date.replace(second=0, microsecond = 0)
            for entry in data_list :
                total = 0
                total_production = sum(entry['production'].values())
                for prod in entry['production'] :
                    total += entry['production'][prod]*self.get_value_for_data(prod)
                self.carbon += total/total_production
                self.carboncount += 1
        except KeyError :
            #print("La clé API ne fonctionne pas, valeur par défaut générée. Si vous voulez un résultat précis, générez une clée sur https://opendata.reseaux-energies.fr/, et définissez la pour RESEAUX_ENERGIES_TOKEN")
            return self.get_value_for_data("France")
        except Exception :
            #print("La clé API ne fonctionne pas, valeur par défaut générée. Si vous voulez un résultat précis, générez une clée sur https://opendata.reseaux-energies.fr/, et définissez la pour RESEAUX_ENERGIES_TOKEN")
            return self.get_value_for_data("France")
    
    def carbonIntensity_date(self, dateti):
        try :
            if (dateti.minute<30) :
                 date = dateti.replace(minute=0)
            else :
                 date = dateti.replace(minute=30)
            data_list = FR.fetch_production()
            date = date.replace(second=0, microsecond = 0)
            for entry in data_list :
                last = entry['datetime'].replace(tzinfo=None)
                if (((last-date).total_seconds()>=0) or entry==data_list[-1]):
                    total = 0
                    total_production = sum(entry['production'].values())
                    for prod in entry['production'] :
                        total += entry['production'][prod]*self.get_value_for_data(prod)
                    self.carbon += total/total_production
                    self.carboncount += 1
        except KeyError :
            #print("La clé API ne fonctionne pas, valeur par défaut générée. Si vous voulez un résultat précis, générez une clée sur https://opendata.reseaux-energies.fr/, et définissez la pour RESEAUX_ENERGIES_TOKEN")
            return self.get_value_for_data("France")
        except Exception :
            #print("La clé API ne fonctionne pas, valeur par défaut générée. Si vous voulez un résultat précis, générez une clée sur https://opendata.reseaux-energies.fr/, et définissez la pour RESEAUX_ENERGIES_TOKEN")
            return self.get_value_for_data("France")

    def nvidia_smi(self, username):
        try:
            usage = 0
            # Exécute la commande pgrep pour obtenir les PID de l'utilisateur spécifié
            pgrep_output = subprocess.check_output(["pgrep", "-u", username]).decode()

            # Divise la sortie en lignes (un PID par ligne)
            pids = [int(pid) for pid in pgrep_output.strip().split('\n')]
            
            nvidia_smi_output = subprocess.check_output(["nvidia-smi", "pmon", "-c", "1", "-s", "um"]).decode()
            lines = nvidia_smi_output.strip().split('\n')
            for line in lines[2:]:  # Omettre les deux premières lignes de commentaires
                tokens = line.split()
                if (tokens[1]== "-"):
                    usage+= 0
                elif int(tokens[1]) in pids :
                    usage += float(tokens[3])
            return usage
        except Exception:
            if self.count==0 :
                print("No NVIDIA graphic card found, GPU utilization is set to 0")
            return 0
    





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
    pue = 1.67
    if len(sys.argv) < 2:
        print("Linux command is missing")
        sys.exit(1)
    for i in sys.argv:
        if (i.startswith("pue=")) :
            pue = sys.argv[2].split('=')[1]
            pue = float(pue.replace(',', '.'))
    monitor = Monitor(10, pue)
    print("Start measuring...")
    linux_command(sys.argv[1], monitor)
    b = monitor.stop()
    if b :
        print("Measurement completed, see the result in the co2 folder")
    else :
        print("Your command could not be executed")

