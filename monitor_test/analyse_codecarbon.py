import sys
import math
import numpy as np
import matplotlib.pyplot as plt

def analyse() :
    lines = open ("co2\log.log", "r").readlines()
    prog = [line.split() for line in lines]
    liste_wh = []
    liste_temps=[]
    count = 0


    for pc in range(len(lines)) :
        if (prog[pc][5]) == "kWh":
            val = float(prog[pc][4])*1000
            liste_wh.append(val)
            count +=1
            liste_temps.append(count*10)
        #else :
            #print(prog[pc])


    #print (prog[19][5])
    plt.plot(liste_temps, liste_wh)
    plt.title("Energie en fonction du temps")
    plt.xlabel("Temps en secondes")
    plt.ylabel("Energie totale en Wh")
    plt.show()