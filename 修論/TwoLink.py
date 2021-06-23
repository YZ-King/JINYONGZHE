import  numpy as np


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



def VGFMatrix(theta=np.array([2,1]),thetadot=np.array([2,1])):

    V1=-m2*l1*r2*(2*thetadot[0]*thetadot[1]+pow(thetadot[1],2))*np.sin(theta[1])
    V2=m2*l1*r2*pow(thetadot[0],2)*np.sin(theta[1])
    V=np.array([V1,V2])
    G1=m1*g*r1*np.cos(theta[0])+m2*g*l1*np.cos(theta[0])+m2*g*r2*np.cos(theta[0]+theta[1])
    G2=m2*g*r2*np.cos(theta[0]+theta[1])
    G=np.array([G1,G2])
    F=np.array([b1*thetadot[0],b2*thetadot[1]])
    VGF=V+G+F
    return (VGF)



def MMatrix(theta=np.array([2,1])):

    M11=m1*pow(r1,2)+m2*pow(l1,2)+m2*pow(r2,2)+I1+I2+2*m2*l1*r2*np.cos(theta[1])
    M12=m2*pow(r2,2)+I2+m2*l1*r2*np.cos(theta[1])
    M21=m2*pow(r2,2)+I2+m2*l1*r2*np.cos(theta[1])
    M22=m2*pow(r2,2)+I2
    M=[[float(M11),float(M12)],[float(M21),float(M22)]]
    return (M)


def T(q=np.array([2,1]),dq=np.array([2,1]),ddq=np.array([2,1])):
    T=np.dot(MMatrix(q),ddq)+VGFMatrix(q,dq)
    return (T)

def main():
    q=np.array([[np.pi/4],[np.pi/4]])   #q1:[o,pi/2] q2:[-pi,pi]
    dq=np.array([[np.pi/2],[np.pi/2]])          #dq1:[-pi/2,pi/2] dq2:[-pi/2,pi/2]
    ddq=np.array([[np.pi*np.pi],[np.pi*np.pi]])         #ddq1:[-pi/2,pi/2] dq2:[-pi/2,pi/2]

    print(MMatrix(q))
    print(VGFMatrix(q,dq))
    print(T(q,dq,ddq))

if __name__ == '__main__':
    main()
