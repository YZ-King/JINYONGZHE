import numpy as np
import TwoLink as TL
import matplotlib.pyplot as plt

m1=0.6
m2=0.7
g=9.80665
l1=0.2
l2=0.3
r1=l1/2
r2=l2/2
I1=1/12*m1*pow(l1,2)
I2=1/12*m2*pow(l2,2)
b1=0.1
b2=0.15
delta_t=0.01
#目標軌道
t=np.linspace(0,5,501)
#0-1s匀加速运动 pi/2 rad/s^2
t1=t[0:101]
ddtheta1=np.pi/2*np.ones(101)
dtheta1=np.pi/2*t1
theta1=1/2*np.pi/2*(t1*t1)
#1s-4s匀速运动
ddtheta2=np.zeros(300)
dtheta2=np.pi/2*np.ones(300)
theta2=np.pi/4*np.ones(300)+dtheta2*np.linspace(0.01,3,300)
#4s-5s匀减速运动
ddtheta3=-np.pi/2*np.ones(100)
dtheta3=np.pi/2*(np.ones(100)-np.linspace(0.01,1,100))
theta3=np.pi*7/4*np.ones(100)+np.pi/2*np.linspace(0.01,1,100)-1/2*np.pi/2*(np.linspace(0.01,1,100)*np.linspace(0.01,1,100))
# 整理する
ddtheta=np.append(ddtheta1,ddtheta2)
ddtheta=np.append(ddtheta,ddtheta3)
dtheta=np.append(dtheta1,dtheta2)
dtheta=np.append(dtheta,dtheta3)
theta=np.append(theta1,theta2)
theta=np.append(theta,theta3)


xr=0.3+0.1*np.sin(theta)
yr=0.1+0.1*np.cos(theta)
dx_dt=0.1*np.cos(theta)*dtheta
dy_dt=-0.1*np.sin(theta)*dtheta
ddx_dt=0.1*(dtheta*(-np.sin(theta))+ddtheta*np.cos(theta))
ddy_dt=0.1*(dtheta*np.cos(theta)+ddtheta*np.sin(theta))
# ddX=np.zeros((2,1,501))
# print(np.array([[dx_dt[0]],[dy_dt[0]]]))
# for i in range(501):
#     ddX[:,:,i]=[[ddx_dt[i]],
#                 [ddy_dt[i]]]
# print(ddX.shape)
#逆運動学
q1=np.arccos((xr*xr+yr*yr+l1*l1-l2*l2)/(2*l1*np.sqrt(xr*xr+yr*yr)))+np.arctan(yr/xr)
q2=np.arctan((yr-l1*np.sin(q1))/(xr-l1*np.cos(q1)))-q1
q=np.zeros((501,2,1))
for i in range(501):
    q[i,:,:]=[[q1[i]],
              [q2[i]]]

#ヤコビ行列
J=np.zeros((501,2,2))
for i in range(501):
    J[i,:,:]=np.array([[-l1*np.sin(q1[i])-l2*np.sin(q1[i]+q2[i]),-l2*np.sin(q1[i]+q2[i])],
            [l1*np.cos(q1[i])+l2*np.cos(q1[i]+q2[i]),l2*np.cos(q1[i]+q2[i])]])

J_INV=np.zeros((501,2,2))
dq_dt=np.zeros((501,2,1))

for i in range(501):
    J_INV[i,:,:]=np.linalg.inv(J[i,:,:])
    dq_dt[i,:,:]=np.dot(J_INV[i,:,:],np.array([[dx_dt[i]],[dy_dt[i]]]))

dq1_dt=dq_dt[:,0,0]
dq2_dt=dq_dt[:,1,0]

J_dt=np.zeros((501,2,2))
for i in range(501):
    J_dt[i,:,:]=np.array([[-l1*np.cos(q1[i])*dq1_dt[i]-l2*np.cos(q1[i]+q2[i])*(dq1_dt[i]+dq2_dt[i]),-l2*np.cos(q1[i]+q2[i])*(dq1_dt[i]+dq2_dt[i])],
            [-l1*np.sin(q1[i])*dq1_dt[i]-l2*np.sin(q1[i]+q2[i])*(dq1_dt[i]+dq2_dt[i]),-l2*np.sin(q1[i]+q2[i])*(dq1_dt[i]+dq2_dt[i])]])

ddq_dt=np.zeros((501,2,1))
for i in range(501):
    ddq_dt[i,:,:]=np.dot(J_INV[i,:,:],(np.array([[ddx_dt[i]],[ddy_dt[i]]])-np.dot(J_dt[i,:,:],dq_dt[i,:,:])))

ddq1_dt=ddq_dt[:,0,:]
ddq2_dt=ddq_dt[:,1,:]

T_d=np.zeros((501,2,1))
for i in range(501):
    T_d[i,:,:]=TL.T(q[i,:,:],dq_dt[i,:,:],ddq_dt[i,:,:])

#实际的角度，角速度，角加速度定义为r,dr,ddr,输入为u
u=np.zeros((501,2,1))
r=np.zeros((501,2,1))
dr=np.zeros((501,2,1))
ddr=np.zeros((501,2,1))

r[0,:,:]=[[np.pi/2],[-np.pi/2]]
dr[0,:,:]=[[0],[0]]

u=T_d
for i in range(501):
    T=u
    M=TL.MMatrix(r[i,:,:])
    VGF=TL.VGFMatrix(r[i,:,:],dr[i,:,:])
    ddr[i,:,:]=np.dot(np.linalg.inv(M),T[i,:,:]-VGF)
    if i<500:
        dr[i+1,:,:]=dr[i,:,:]+ddr[i,:,:]*delta_t
        r[i+1,:,:]=r[i,:,:]+dr[i,:,:]*delta_t

#   順運動学
x=l1*np.cos(r[:,0,0])+l2*np.cos(r[:,0,0]+r[:,1,0])
y=l1*np.sin(r[:,0,0])+l2*np.sin(r[:,0,0]+r[:,1,0])

print(x.shape)

plt.figure()
plt.plot(x,y)
plt.plot(xr,yr)
plt.show()


