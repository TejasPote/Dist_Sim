import numpy as np
from utils import hydrau, entliq, entvap, compute_avg, bubblepoint
import pandas as pd
# import json 
from tqdm import tqdm

 

def model(data, xb, xd, x, m, R, Q, TF, itr, save):

    num_iter = itr
    dt =  data['loop']['dt']
    NT = data['column']['NT']
    NF = data['column']['NF']

    eff = data['column']['eff']
    W_l = data['column']['W_l']
    W_h_s = data['column']['W_h_s']
    W_h_r = data['column']['W_h_r']

    d_col = data['column']['d_col']

    # x = np.zeros(NT + 2)
    # m = np.zeros(NT + 2)
    y = np.zeros(NT + 1)
    T = np.zeros(NT + 1)
    Hl = np.zeros(NT + 1)
    Hv = np.zeros(NT + 1)
    L = np.zeros(NT + 1)
    V = np.zeros(NT + 1) 
    dm = np.zeros(NT + 1)
    dxm = np.zeros(NT + 1)

    # for i in range(1, NT+1):
    #     x[i] = data['tray']['x_'+str(i-1)]
    #     m[i] = data['tray']['m_'+str(i-1)]

    # xb = data['tray']['xb']
    # xd = data['tray']['xd']
    mb = data['flow']['mb']
    md = data['flow']['md']

    P = data['flow']['P']
    # Qr = data['flow']['Qr']
    F = data['flow']['F']
    xF = data['flow']['xF']
    # Vb = data['flow']['Vb']
    # TF = data['flow']['TF']
    # R = data['flow']['R']


    time = 0.00

    cols = ['Time','Vb','Tb','xb','Td','xd','D','B','md','mb','vb']
    cols.extend(['x_'+str(i+1) for i in range(NT)])
    cols.extend(['T_'+str(i+1) for i in range(NT)])
    cols.extend(['m_'+str(i+1) for i in range(NT)])
    cols.extend(['L_'+str(i+1) for i in range(NT)])

    logs = pd.DataFrame(columns=cols)

    xm = np.zeros(NT+1)
    xm =  x*m

    for j in tqdm(range(num_iter)):
            
        
        # ***********-------->Vapour phase compositions and enthalpy calculations<--------***********

        #Reboiler

        yb,_,Tb = bubblepoint(xb, P)
        Hlb = entliq(Tb,xb)
        Hvb = entvap(Tb,yb)
        
        # 1st tray 

        y[1],_,T[1] = bubblepoint(x[1], P)
        y[1] = yb + eff*(y[1] - yb)
        Hl[1] = entliq(T[1], x[1])
        Hv[1] = entvap(T[1], y[1])

        # 2nd to NT trays
        for i in range(2, NT+1):
            y[i], _, T[i] = bubblepoint(x[i], P)
            y[i] = y[i-1] + eff*(y[i] - y[i-1])
            Hl[i] = entliq(T[i], x[i])
            Hv[i] = entvap(T[i], y[i])

        # Reflux drum

        yd, _, Td = bubblepoint(xd, P)
        Hld = entliq(Td, xd)

        # ***********-------->Internal liquid flow rate calculations<--------***********

        for i in range(1,NF+1):
            L[i] = hydrau(m[i], x[i], W_h_s, W_l, d_col)
        
        for i in range(NF+1, NT+1):
            L[i] = hydrau(m[i], x[i], W_h_r, W_l, d_col)

        # ***********-------->Vapour Flow rate calculations<--------***********
        

        # Reboiler vapour flow rate
        # print("L1:", L[1])
        Vb = (Q-L[1]*(Hlb-Hl[1]))/(Hvb-Hlb)
        # print("Vb:", Vb) 
        B = L[1]-Vb

    
        # print(Vb)
        # print(B)
        if B<0:
            print("Early stopping due to negative bottoms flow rate..")
            break

        # 1st tray
        V[1]=(Hl[2]*L[2]+Hvb*Vb-Hl[1]*L[1])/Hv[1]


        # 2nd to feed tray
        for i in range(2,NF):
            V[i]=(Hl[i+1]*L[i+1]+Hv[i-1]*V[i-1]-Hl[i]*L[i])/Hv[i]


        #feed tray
        Hf = entliq(TF, xF)
        V[NF]=(Hl[NF+1]*L[NF+1]+Hv[NF-1]*V[NF-1]-Hl[NF]*L[NF]+Hf*F)/Hv[NF]

        # (NF+1)th tray to (NT-1)th tray

        for i in range(NF+1,NT):
            V[i]=(Hl[i+1]*L[i+1]+Hv[i-1]*V[i-1]-Hl[i]*L[i])/Hv[i]

        # NTth tray
            
        V[NT] = (Hld*R+Hv[NT-1]*V[NT-1]-Hl[NT]*L[NT])/Hv[NT] 
        D = V[NT] - R 
        if D<0:
            print("Early stopping due to negative distillate flow rate..")
            break

        ## ***********-------->Evaluating time derivatives<--------***********

        # 1st tray
        dm[1] = L[2]+Vb-V[1]-L[1]

        # 2nd to (NF-1)th tray
        for i in range(2, NF):
            dm[i] = L[i+1]+V[i-1]-L[i]-V[i]

        # Feed tray 
        dm[NF]=L[NF+1]+F+V[NF-1]-L[NF]-V[NF]

        #(NF+1)th to (NT-1)th tray
        for i in range(NF+1, NT):
            dm[i]=L[i+1]+V[i-1]-L[i]-V[i]

        # NTth tray
        dm[NT]=R+V[NT-1]-L[NT]-V[NT]

        # Reboiler

        dxb = (x[1]*L[1]-yb*Vb-xb*B)/mb

        # 1st tray
        dxm[1] = x[2]*L[2]+yb*Vb-x[1]*L[1]- y[1]*V[1]

        # 2nd to (NF-1)th tray
        for i in range(2, NF):
            dxm[i] = x[i+1]*L[i+1]+y[i-1]*V[i-1]-x[i]*L[i]- y[i]*V[i]

        # Feed tray 
        dxm[NF]=x[NF+1]*L[NF+1]+y[NF-1]*V[NF-1]-x[NF]*L[NF]-y[NF]*V[NF]+F*xF

        #(NF+1)th to (NT-1)th tray
        for i in range(NF+1, NT):
            dxm[i] = x[i+1]*L[i+1]+y[i-1]*V[i-1]-x[i]*L[i]- y[i]*V[i]

        # NTth tray
        dxm[NT]=xd*R + y[NT-1]*V[NT-1] - x[NT]*L[NT] - y[NT]*V[NT]


        # Condenser
        dxd = (V[NT]*y[NT]-(R+D)*xd)/md

        # ***********-------->Euler's numerical integration<--------***********

        m = m + dm*dt

        xb = xb + dxb*dt
        if(xb<0.00):
            xb=0.00
        if(xb>1.00):
            xb=1.00
            
        xm = xm + dxm*dt
        for i in range(1,NT+1):
            x[i] = xm[i]/m[i]

        for i in range(1,NT+1):
            if(x[i]<0.00):
                x[i]=0.00
            if(x[i]>1.00):
                x[i]=1.00


        xd= xd + dxd*dt
        if(xd<0.00):
            xd=0.00
        if(xd>1.00):
            xd=1.00
        
        logs.loc[j] = np.hstack((time, Vb, Tb, xb, Td, xd, D, B, md, mb, Vb, x[1:NT+1], T[1:NT+1],m[1:NT+1],L[1:NT+1]))
        time += dt
    
    if save:
        logs.to_csv("steady_state.csv")
    return (xb, xd, x, m)