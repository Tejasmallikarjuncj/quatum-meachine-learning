#import the required modules 
import matplotlib.pyplot as plt
import pandas as pd
import math as m
from qiskit import *

fig, ax = plt.subplots()
ax.set(xlabel='Data Feature 1', ylabel='Data Feature 2')

# Get the data from the .csv file
data = pd.read_csv('kmeans_data.csv',
    usecols=['Feature 1', 'Feature 2', 'Class'])

# Create binary variables to filter data
isGreen = data['Class'] == 'Green'
isBlue = data['Class'] == 'Blue'
isBlack = data['Class'] == 'Black'

# Filter data
greenData = data[isGreen].drop(['Class'], axis=1)
blueData = data[isBlue].drop(['Class'], axis=1)
blackData = data[isBlack].drop(['Class'], axis=1)

# Making the x-coords points of data and y-coords of data
x_g = list(greenData['Feature 1'])
x_b = list(blueData['Feature 1'])
x_k = list(blackData['Feature 1'])
y_g = list(greenData['Feature 2'])
y_b = list(blueData['Feature 2'])
y_k = list(blackData['Feature 2'])

# This is the point we need to classify
y_p = 0.0
x_p = 0.0

# Finding the x-coords of the centroids
xgc = sum(x_g)/len(x_g)
xbc = sum(x_b)/len(x_b)
xkc = sum(x_k)/len(x_k)

# Finding the y-coords of the centroids
ygc = sum(y_g)/len(y_g)
ybc = sum(y_b)/len(y_b)
ykc = sum(y_k)/len(y_k)

# Plotting the centroids
plt.plot(xgc, ygc, 'gx')
plt.plot(xbc, ybc, 'bx')
plt.plot(xkc, ykc, 'kx')

# Plotting the new data point
plt.plot(x_p, y_p, 'ro')

# Setting the axis ranges
plt.axis([-1, 1, -1, 1])

plt.show()

#calculating theta and phi value
phi_values = [(x + 1)*m.pi/2 for x in [x_p,xgc,xbc,xkc]]
theta_values = [(y + 1)*m.pi/2 for y in [x_p,ygc,ybc,ykc]]
class_list = ['Green','Blue','Black']
#creating quantum circuit for the k-mean distance
k_cir = QuantumCircuit(3,3)
count_result = {}
for i in range(1,4):
    #intializing the data and centroid qubits
    k_cir.h(0)
    k_cir.u3(theta_values[0],phi_values[0],0,1)
    k_cir.u3(theta_values[i],phi_values[i],0,2)
    k_cir.barrier()
    
    #applying the controlled swap to data and centroid qubit
    k_cir.cswap(0,1,2)
    k_cir.h(0)
    k_cir.barrier()
    
    #measuring the qubit and resetting
    k_cir.measure([0],[0])
    k_cir.reset(0)
    k_cir.reset(1)
    k_cir.reset(2)
    
    #executing the circuit
    counts = execute(k_cir, backend = Aer.get_backend('qasm_simulator'), shots=1024).result().get_counts()
    count_result[class_list[i-1]] = counts['001']

    
#let's find the class the point belongs to    
for classes,dist in list(count_result.items()):
    if min(list(count_result.values())) == dist:
        print("From Quantum distance estiamtion the point ({}, {}) is in the class {}".format(x_p,y_p,classes))
        break

#let"s verify
classical_result = {}
x_point = [xgc,xbc,xkc]
y_point = [ygc,ybc,ykc]
for i in range(0,3):
    distance = m.sqrt((x_point[i]-x_p)**2 + (y_point[i]-y_p)**2)
    classical_result[class_list[i]] = distance

#let's find the class the point belongs to    
for classes,dist in list(classical_result.items()):
    if min(list(classical_result.values())) == dist:
        print("From Classical distance estiamtion the point ({}, {}) is in the class {}".format(x_p,y_p,classes))
        break

