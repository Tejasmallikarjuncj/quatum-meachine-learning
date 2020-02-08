#importing required modules
from qiskit import *
from qiskit.tools.visualization import *
import numpy as np
import matplotlib.pyplot as plt

#creating classical_regression model
class linear_regression():
    def __init__(self,x,y):
        self.x = x
        self.y = y
    
    #partial derivative of m
    def diff_m(self,m,c):
        val_c = np.ones(self.x.size,dtype=int).transpose()*c
        return np.dot(np.array([np.ones(self.x.size,dtype=int)]),2*self.x*(m*self.x+val_c-self.y))[0][0]/len(list(self.x))
    
    #partial derivative of c
    def diff_c(self,m,c):
        val_c = np.ones(self.x.size,dtype=int).transpose()*c
        return np.dot(np.array([np.ones(self.x.size,dtype=int)]),2*(m*self.x+val_c-self.y))[0][0]/len(list(self.x))
    
    #naive gradient_decent
    def gradient_decent(self, n):#n determines the precision of slope of the line
        m = 1.0
        c = 0.0
        for i in range(n):
            m = m - 0.000005*self.diff_m(m,c)
            #c = c - 0.0015*self.diff_c(m,c)
            if self.diff_m(m,c) <= 0.0000001 and self.diff_m(m,c)>= -0.0000001:
                break
        return m,c

#taking data
data = np.array([[i*0.5 for i in range(1,14)]]).transpose()
val = np.array([[0.5, 1.5, 2.25, 3.0, 3.5, 4.0, 4.75, 5.00, 5.5, 6.0, 6.5, 7.0, 7.75]]).transpose()
line = linear_regression(data,val)

#generating population, where 1.5453 is the optimized value
theta = {'optimised':2*0.889}

#keeping track of samples
sample = {'s1':[],'s2':[],'s3':[],'s4':[],'s5':[],'s6':[]}
fitt = {'s1':[],'s2':[],'s3':[],'s4':[],'s5':[],'s6':[]}

#The population size is 6
for i in range(1,7):
    m,c = line.gradient_decent(i*10)
    theta['samp'+str(i)] = round(2*np.arctan(round(m,3)),3)    
    sample['s'+str(i)].append((round(m,3)))
    
#creating quantum genetic sub circuit

#crossover circuit
crossover_circuit = QuantumCircuit(2, name='Crossover')
crossover_circuit.cu3(0.05*(np.pi)/2,0,0,0,1)
crossover_circuit.cu3(0.05*(np.pi)/2,0,0,1,0)
crossover = crossover_circuit.to_instruction()

#mutation circuit
mutation_circuit = QuantumCircuit(1, name='mutation')
mutation_circuit.u3(0.001*(np.pi)/2,0,0,0)
mutation = mutation_circuit.to_instruction()

#fittness circuit
fittness_circuit = QuantumCircuit(3,1,name='fitness')
fittness_circuit.h(2)
fittness_circuit.cswap(2,1,0)
fittness_circuit.h(2)
fittness_circuit.measure([2],[0])
fittness = fittness_circuit.to_instruction()

#new_generation circuit
gen_circuit = QuantumCircuit(3,name="new_generation")
gen_circuit.h(2)
gen_circuit.cswap(2,1,0)
gen_circuit.h(2)
new_generation = gen_circuit.to_instruction()

#applying the aglorithm
for i in range(1,7,2): 
    for j in range(1,25):    
        #creating quantum circuit taking two samples at a time
        qregs = QuantumRegister(5,name='q')
        cregs = ClassicalRegister(4,name='c')
        sim = Aer.get_backend('qasm_simulator')
        gene_circuit = QuantumCircuit(qregs,cregs,name='Genetic_algorithm')

        #initizalizing the qubit
        gene_circuit.u3(theta['optimised'],0,0,qregs[0])
        gene_circuit.u3(theta['samp'+str(i)],0,0,qregs[1])
        gene_circuit.u3(theta['samp'+str(i+1)],0,0,qregs[2])
        gene_circuit.barrier()

        #applying genetic algorithm
        gene_circuit.append(crossover,[qregs[1],qregs[2]])
        gene_circuit.append(mutation,[qregs[1]])
        gene_circuit.append(mutation,[qregs[2]])
        gene_circuit.append(fittness,[qregs[0],qregs[1],qregs[3]],[cregs[0]])
        count = execute(gene_circuit,backend=sim,shots=6000).result().get_counts()
        fitt['s1'].append(count['0001']/sum(list(count.values())))
        gene_circuit.append(new_generation,[qregs[0],qregs[1],qregs[3]])
        gene_circuit.append(fittness,[qregs[0],qregs[2],qregs[4]],[cregs[1]])
        count_1 = execute(gene_circuit,backend=sim,shots=6000).result().get_counts()
        fitt['s2'].append((count_1['0011'])/sum(list(count_1.values())))
        gene_circuit.append(new_generation,[qregs[0],qregs[2],qregs[4]])

    gene_circuit.measure([qregs[1],qregs[2]],[cregs[2],cregs[3]])
    count_2 = execute(gene_circuit,backend=sim, shots = 6000).result().get_counts()
    p1 = (count_2['1100']+count_2['1001']+count_2['1000'])/sum(list(count_2.values()))
    p2 = (count_2['0101']+count_2['1100']+count_2['0100'])/sum(list(count_2.values()))
    sample['s'+str(i)].append(round(np.tan(round(np.arcsin(np.sqrt(p1)),3)),3))
    sample['s'+str(i+1)].append(round(np.tan(round(np.arcsin(np.sqrt(p2)),3)),3))
    gene_circuit.reset(qregs)
    
#the final optimized sample are
print("The sample",sample)
print("the optimium solution", 0.867)

#ploting for the graph of data

#plotting for the graph for optimized parameter

val_c = np.array([np.zeros(data.size,dtype=int)]).transpose()
val_y = data*1.227+val_c
plt.scatter(list(data.transpose()), list(val.transpose()), marker='x', color = 'r', label='data points')
plt.plot(data,val_y, color='r', label='optimized')

#plotting the graph for the samples
for i in range(1,7):
    val_ysamp = data*sample['s'+str(i)][-1]+val_c
    plt.plot(data,val_ysamp, color='b', label='sample'+str(i))
plt.title("best_fit_line")
plt.xlabel('x_data')
plt.ylabel('y_data')
plt.legend()
plt.grid(True)
plt.show()   
