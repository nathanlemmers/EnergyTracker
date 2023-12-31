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
import subprocess
import sys
from datetime import datetime, timedelta
import pandas as pd
import FR
import warnings
warnings.filterwarnings("ignore")


class Monitor(Thread):
    global total_cpu, count, cpu_now, total_gpu, gpu_now,  tracker_codecarbon, tracker_eco2ai, tracker_carbontracker, total_time, start_time, total_ram,num_gpus, pid, dossier, carbon, carboncount, new_time, pue, GPU_power_info

    def __init__(self, delay, pue):
        self.pue = pue
        super(Monitor, self).__init__()
        #Création du dossier co2 pour ranger les données, un sous dossier ave la date sera fait ensuite.
        rep="co2"
        if (not(os.path.exists(rep))):
            os.mkdir(rep)
        #Variable pour continuer le while du run.
        self.stopped = False
        #delay pour faire les mesures, dans l'executable il est de 10s, si on augmente on risque de consommer plus, et si on diminue, on aura des mesures d'usage factor peu fiable
        self.delay = delay 
        self.total_cpu = 0
        self.count = 0 
        self.total_gpu = 0
        self.GPU_power_info = 0
        self.total_ram = 0
        self.carboncount = 0
        self.carbon = 0
        self.new_time = datetime.now()
        #sous dossier :
        self.dossier = self.new_time.strftime("%Y.%m.%d_%H:%M:%S")
        if (not(os.path.exists("co2/"+self.dossier))):
            os.mkdir("co2/"+self.dossier)
        #Initialisation de CodeCarbon
        from codecarbon import EmissionsTracker
        self.tracker_codecarbon = EmissionsTracker(output_dir='co2/'+self.dossier, output_file='codecarbon_emissions.csv', tracking_mode = 'process')
        
        #Initialisation de eco2ai
        self.tracker_eco2ai = eco2ai.Tracker(project_name="My_default_project_name", experiment_description="We trained...", file_name="co2/"+self.dossier+"/eco2ai_emissions.csv")
        
        #Initialisation de CarbonTracker 
        #self.tracker_carbontracker = CarbonTracker(epochs=1, log_dir="co2/"+self.dossier, log_file_prefix="carbontracker_emissions")
        

        # Initialiser la bibliothèque pynvml
        try : 
            pynvml.nvmlInit()
            self.num_gpus = pynvml.nvmlDeviceGetCount()
            self.gpu_ref = []
            for i in range(self.num_gpus) :
                #Tableau de référence pour la consommation passive de nos GPUs
                self.gpu_ref.append(pynvml.nvmlDeviceGetPowerUsage(pynvml.nvmlDeviceGetHandleByIndex(i)))

        except pynvml.NVMLError_FunctionNotFound:
            # La fonction n'est pas trouvée, la carte graphique Nvidia n'est pas disponible
            self.num_gpus = 0
        except pynvml.NVMLError as e:
            # Une autre erreur liée à pynvml s'est produite
            self.num_gpus = 0
        self.start_time = time.time()

        #Lancement des modules externes
        self.tracker_codecarbon.start()
        self.tracker_eco2ai.start()
        #self.tracker_carbontracker.epoch_start()
        self.pid = os.getpid()
        
        self.start()

    def run(self):
        #Premier delay pour attendre le lancement du code principal
        time.sleep(self.delay)
        current_user = psutil.Process().username()
        while not self.stopped:
            loop = time.time()
            #Ici, on cherche la consommation du GPU pour GA dapated. Donc on lance prend la mesure toutes les delay et on prendra ensuite la moyenne.
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
                            if self.gpu_now_pynvml==0 :
                                self.gpu_ref[i] = pynvml.nvmlDeviceGetPowerUsage(handle)
                            gpu_mwatt = pynvml.nvmlDeviceGetPowerUsage(handle) -self.gpu_ref[i]
                            if gpu_mwatt<0 :
                                gpu_mwatt=0
                            break
            except pynvml.NVMLError_FunctionNotFound:
            # La fonction n'est pas trouvée, la carte graphique Nvidia n'est pas disponible
                pass
            except pynvml.NVMLError as e:
                # Une autre erreur liée à pynvml s'est produite
                pass
            #nvidia-smi donne l'usage factor de nos GPUs lié à notre user seulement
            gpu_now = self.nvidia_smi(current_user)
            self.GPU_power_info += gpu_mwatt
            self.total_gpu += gpu_now
            #On mesure l'usage factor du CPU lié seulement à notre user courant
            user_processes = [p.info for p in psutil.process_iter(['pid', 'username', 'cpu_percent']) if p.info['username'] == current_user]
            cpu_total = sum(p['cpu_percent'] for p in user_processes)
            cpu_cores = psutil.cpu_count(logical=False)
            cpu_usage = cpu_total / cpu_cores if cpu_cores > 0 else 0
            self.total_cpu += cpu_usage
            #On mesure la ram moyenne utilisé par notre user
            user_processes = [p for p in psutil.process_iter(['pid', 'username']) if p.info['username'] == current_user]
            self.total_ram += sum(p.memory_info().rss / (1024 ** 3) for p in user_processes)
            self.count += 1
            #Cette boucle permet la mesure moyenne de l'intensité carbone si on depasse une journée de travail
            if (self.new_time-datetime.now()>timedelta(hours=23.5)):
                try :
                    self.new_time = datetime.now()
                    self.carbonIntensity_day(self.new_time)
                except Exception :
                    self.carbon += self.get_value_for_data("France")*48
                    self.carboncount += 48 
            time.sleep(self.delay-(time.time()-loop))


    def stop(self):
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
        nb_cpu = psutil.cpu_count(logical=False)
        gpu_name = "No NVIDIA graphic card found"
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
            gpu_usage = self.total_gpu/self.count/100
            gpu_mwatt = self.GPU_power_info/self.count
            GA = self.GreenAlgo_adapted(self.total_time, cpu_name, nb_cpu, average_ram, cpu_usage, gpu_mwatt)
            
            fichier = open("co2/"+self.dossier+"/GreenAlgorithm_emissions.txt", "w")
            if GA is not None :
                fichier.write("Stats :\n{} g de co2.\n{} kWh\n".format(GA[0], GA[1]))
                self.fichier_total_GA(GA[0], GA[1])
            else :
                #On écrit les output à mettre à la main dans GreenAlgorithm ici. 
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

    def GreenAlgo_adapted(self, time_code, cpu_name , number_core_code, ram_moyenne, CPU_usage, GPU_mwatt) :
        runTime=time_code
        numberCPUs=number_core_code
        CPUpower = self.get_value_for_data(cpu_name)
        if CPUpower is not None :
            memory=ram_moyenne
            power_memory = self.get_value_for_data("memory_power")
            usageCPU=CPU_usage
            PSF=1
            PUE_used = self.pue
            powerNeeded_CPU = PUE_used * numberCPUs * CPUpower * usageCPU
            powerNeeded_GPU = PUE_used * GPU_mwatt / 1000
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
    #Donne les valeurs que l'on recherche dans data_GA
    def get_value_for_data(self, target_data):
        df = pd.read_csv("Monitor/data_GA.csv")
        row = df[df['data'] == target_data]
        if not row.empty:
            return float((row['value'].values[0]).replace(',','.'))
        return None

    def changepid(self, pid):
        self.pid=pid

    #Donne l'intensité carbone moyenne de la dernière journée
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
    
    #Donne l'intensité carbone à partir de la date en argument
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


    #Donne l'usage factor du GPU lié à l'user en argument
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
    





#Pour lsancer la commande linux
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
    freq=10
    pue = 1.67
    if len(sys.argv) < 2:
        print("Linux command is missing")
        sys.exit(1)
    for i in sys.argv:
        if (i.startswith("pue=")) :
            pue = i.split('=')[1]
            pue = float(pue.replace(',', '.'))
        if (i.startswith("freq=")) :
            freq = i.split('=')[1]
            freq = float(freq.replace(',', '.'))
    monitor = Monitor(freq, pue)
    print("Start measuring...")
    linux_command(sys.argv[1], monitor)
    b = monitor.stop()
    if b :
        print("Measurement completed, see the result in the co2 folder")
    else :
        print("Your command could not be executed")

