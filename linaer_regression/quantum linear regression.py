import numpy as np
import matplotlib.pyplot as plt

class linaer_regression():
    def __init__(self,x,y):
        self.x = x
        self.y = y
        
    def diff_m(self,m,c):
        val_c = np.ones(self.x.size,dtype=int).transpose()*c
        return np.dot(np.array([np.ones(self.x.size,dtype=int)]),2*self.x*(m*self.x+val_c-self.y))[0][0]/len(list(self.x))
    
    def diff_c(self,m,c):
        val_c = np.ones(self.x.size,dtype=int).transpose()*c
        return np.dot(np.array([np.ones(self.x.size,dtype=int)]),2*(m*self.x+val_c-self.y))[0][0]/len(list(self.x))
    
    def gradient_decent(self):
        m = 1.5
        c = 0.0
        flag = 0
        while flag == 0:
            m = m - 0.000005*self.diff_m(m,c)
            #c = c - 0.0015*self.diff_c(m,c)
            if self.diff_m(m,c) <= 0.0000001 and self.diff_m(m,c)>= -0.0000001:
                flag = 1
        return m,c

#taking data
data = np.array([[i*0.2 for i in range(1,14)]]).transpose()
val = np.array([[1.0, 4.0, 6.0, 9.0, 11.0, 14.0, 16.0, 19.0, 22.0, 24.0, 27.0, 30.0, 32.0]]).transpose()
line = linear_regression(data,val)

#generating population, where 1.5453 is the optimized value
theta = {'optimised':2*1.48}

#keeping track of samples
sample = {'s1':[],'s2':[],'s3':[],'s4':[],'s5':[],'s6':[]}
fitt = {'s1':[],'s2':[],'s3':[],'s4':[],'s5':[],'s6':[]}

#The population size is 6
for i in range(6,12):
    m,c = line.gradient_decent(i*10)
    theta['samp'+str(i-5)] = round(2*np.arctan(round(m,3)),3)    
    sample['s'+str(i-5)].append(round(np.arctan(round(m,3)),3))
    
    
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
        count = execute(gene_circuit,backend=sim,shots=2048).result().get_counts()
        fitt['s1'].append(count['0001']/sum(list(count.values())))
        gene_circuit.append(new_generation,[qregs[0],qregs[1],qregs[3]])
        gene_circuit.append(fittness,[qregs[0],qregs[2],qregs[4]],[cregs[1]])
        count_1 = execute(gene_circuit,backend=sim,shots=2048).result().get_counts()
        fitt['s2'].append((count_1['0010']+count_1['0011'])/sum(list(count_1.values())))
        gene_circuit.append(new_generation,[qregs[0],qregs[2],qregs[4]])

    gene_circuit.measure([qregs[1],qregs[2]],[cregs[2],cregs[3]])
    count_2 = execute(gene_circuit,backend=sim, shots = 6000).result().get_counts()
    p1 = (count_2['1110']+count_2['1100']+count_2['1111']+count_2['1001']+count_2['1101']+count_2['1000'])/sum(list(count_2.values()))
    p2 = (count_2['0111']+count_2['0101']+count_2['1100']+count_2['1111']+count_2['1101']+count_2['0110'])/sum(list(count_2.values()))
    sample['s'+str(i)].append(np.tan(round(np.arcsin(np.sqrt(p1)),3)))
    sample['s'+str(i+1)].append(np.tan(round(np.arcsin(np.sqrt(p2)),3)))
    gene_circuit.reset(qregs)
    
#the final optimized sample are
print("The sample",sample)
print("the optimium solution", 1.48)
