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

