import TwoLink as tl
import numpy as np
import csv

x_test=[]
y_test=[]
# q1=np.linspace(0,np.pi/2,7)   #首尾数据和train重了，处理一下
# q2=np.linspace(-np.pi,np.pi,9)
# dq1=np.linspace(-np.pi/2,np.pi/2,7)
# dq2=np.linspace(-np.pi/2,np.pi/2,7)
# ddq1=np.linspace(-np.pi/2,np.pi/2,7)
# ddq2=np.linspace(-np.pi/2,np.pi/2,7)

q1=np.random.uniform(0,np.pi/2,size=(10000,))   #首尾数据和train重了，处理一下
print(q1)
q2=np.random.uniform(-np.pi,np.pi,size=(10000,))
dq1=np.random.uniform(-np.pi/2,np.pi/2,size=(10000,))
dq2=np.random.uniform(-np.pi/2,np.pi/2,size=(10000,))
ddq1=np.random.uniform(-np.pi/2,np.pi/2,size=(10000,))
ddq2=np.random.uniform(-np.pi/2,np.pi/2,size=(10000,))

for i in range(10000):

    q=np.array([[q1[i]],[q2[i]]])
    dq = np.array([[dq1[i]], [dq2[i]]])
    ddq=np.array([[ddq1[i]],[ddq2[i]]])
    T=tl.T(q,dq,ddq)
    x_test.append([q1[i],q2[i],dq1[i],dq2[i],ddq1[i],ddq2[i]])
    y_test.append([T[0][0],T[1][0]])

x_test=np.array(x_test)
y_test=np.array(y_test)
print(x_test.shape)
print(y_test.shape)

np.save('x_test.npy',x_test)
np.save('y_test.npy',y_test)