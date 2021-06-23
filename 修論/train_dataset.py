import TwoLink as tl
import numpy as np
import csv

x_train=[]
y_train=[]
q1=np.linspace(0,np.pi/2,10)
q2=np.linspace(-np.pi,np.pi,20)
dq1=np.linspace(-np.pi/2,np.pi/2,10)
dq2=np.linspace(-np.pi/2,np.pi/2,10)
ddq1=np.linspace(-np.pi/2,np.pi/2,10)
ddq2=np.linspace(-np.pi/2,np.pi/2,10)

for a in range(len(q1)):
    for b in range(len(q2)):
        for c in range(len(dq1)):
            for d in range(len(dq2)):
                for e in range(len(ddq1)):
                    for f in range(len(ddq2)):
                        q=np.array([[q1[a]],[q2[b]]])
                        dq = np.array([[dq1[c]], [dq2[d]]])
                        ddq=np.array([[ddq1[e]],[ddq2[f]]])
                        T=tl.T(q,dq,ddq)
                        x_train.append([q1[a],q2[b],dq1[c],dq2[d],ddq1[e],ddq2[f]])
                        y_train.append([T[0][0],T[1][0]])
x_train=np.array(x_train)
y_train=np.array(y_train)
print(x_train.shape)
print(y_train.shape)

np.save('x_train.npy',x_train)
np.save('y_train.npy',y_train)