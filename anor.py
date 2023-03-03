#
#
# Giuseppe Stecca
# giuseppe.stecca@iasi.cnr.it
# Code for paper "A bi-objective model to schedule green investment in a two-stage supply chain"
# the code will run if the instance file "Instanceanor.xlsx" is located in the same folder of this file and a subfolder named results is created
#

import gurobipy as gb
import itertools
import numpy as np
from numpy.linalg import norm
import sys
import json
import time
import pandas as pd
import os
import math

Lmin = 1
Bmin = 1
LT = 5
T = [t for t in range(1,LT+1)] # Time horizon
T0 = [t for t in range(0,LT+1)] # Time horizon
LS = 3
S = [i for i in range(1,LS+1)]
LF = 3
F = [j for j in range(1,LF+1)]
A = [(i,j) for (i,j) in itertools.product(S,F) ]
filenameRes = 'results_LINEAR{}_BINARY{}_LOADSOL{}_S{}_F{}_T{}_B{}_D{}_PhiHat{}_PhiBar{}_RhoBar{}_eta{}_MT{}.xlsx'
flienameToLoad = 'results_LINEAR{}_BINARY{}_LOADSOL{}_S{}_F{}_T{}_B{}_D{}_PhiHat{}_PhiBar{}_RhoBar{}_eta{}_MT{}.xlsx'


np.random.seed(0)  # per ripetibilità test
#Capacity of supplier k in period t
SK =  { (k,t): v for (k,t),v in zip (itertools.product(S,T), [100 for i in range(len(S)*len(T))] ) } # supplier capacity
CJ =  { (k,t): v for (k,t),v in zip (itertools.product(F,T), [100 for i in range(len(F)*len(T))] )}# plant capacity

D = 100* min(LF,LS) # *500 
B = D #*2
phi_hat_0 = 800
phi_bar_0 = 1000
rho_bar_0 = 1
phi_hat = {j:phi_hat_0 for j in F }
phi_bar = {j:phi_bar_0 for j in F}
rho_bar = {k:rho_bar_0 for k in T}
alpha = 0.4
BigM = D*B
eta = 0.5
LB=True
LBValue = 0
LOADSOLUTION = True
MODELTYPE = 1
HEURISTIC_START = True
CHECK_HEUR_FEASIBILITY = True # check the feasibility of the heuristic solution

class Result:
    def __init__(self, lenS, lenF, seed, B, D, bestFound=0.0, worstFound=0.0, 
                    itBestFound=0, timeBestFound=0.0, bestUnfeasible=0.0,
                    worstUnfeasible=0.0, totalTime=0.0, maxSol=0.0) -> None:
        self.lenS : int = lenS
        self.lenF : int = lenF
        self.seed : int = seed
        self.B : int = B
        self.D : int = D
        self.bestFound : float = bestFound
        self.worstFound : float = worstFound
        self.itBestFound : int = itBestFound
        self.timeBestFound : float = timeBestFound
        self.bestUnfeasible : float = bestUnfeasible
        self.worstUnfeasible : float = worstUnfeasible
        self.totalTime : float = totalTime
        self.maxSol : float = maxSol
        self.improvement : float = 0

   
    def save (self, filename):
        df = pd.DataFrame(columns=['S', 'F', 'BD', 'B', 'D', 'seed', 'bestFound', 'worstFound', 
                          'itBestFound', 'timeBestFound', 'bestUnfeasible', 'worstUnfeasible', 'totalTime', 'maxSol', 'improvement'])
        df.loc[0] = [self.lenS, self.lenF, self.B/self.D, self.B, self.D, self.seed, self.bestFound, self.worstFound, 
                     self.itBestFound, self.timeBestFound, self.bestUnfeasible, self.worstUnfeasible, 
                     self.totalTime, self.maxSol, 100*(self.maxSol - self.bestFound)/self.maxSol]
        if os.path.isfile(filename):
            df.to_csv(filename, mode='a', index=None, header=False)
        else:
            df.to_csv(filename, index=None)
        return
class Vars:
    def __init__(self):
        self.x = {}
        self.z = {}
        self.phi = {}
        self.rho = {}
        self.y = {}
        self.Z1 = gb.LinExpr()
        self.Z2 = gb.LinExpr()
        self.xStart = {}
        self.zStart = {}
        self.phiStart = {}
        self.rhoStart = {}
        self.yStart = {}
        self.Z1Ub = BigM
        self.Z2Ub = BigM
        self.ub = BigM
    def initVars(self, x, z, phi, rho, y, Z1, Z2):
        self.x = x
        self.z = z
        self.phi = phi
        self.rho = rho
        self.y = y
        self.Z1 = Z1
        self.Z2 = Z2


def getRhoBar(rdecrease, T):
    # -> dict[int, float]:
    rho_bar = {0:1/rdecrease}
    for i in T:
        rho_bar[i] = rho_bar[i-1]*rdecrease  #1
    return rho_bar

def loadSol(filename):
    with open(filename) as json_file:
        data = json.load(json_file)
        x = {(a['i'],a['j']):a['v'] for a  in data['solution'][0]['x'] }
        z = {a['i']:a['v'] for a  in data['solution'][1]['z'] }
        return x, z

def loadSolExcel(filename):

    dfx = pd.read_excel(filename, sheet_name='x')
    dfz = pd.read_excel(filename, sheet_name='z')
    dfphi = pd.read_excel(filename, sheet_name='phi')
    dfrho = pd.read_excel(filename, sheet_name='rho')
    dfy = pd.read_excel(filename, sheet_name='y')


    x_load = {(row['k'], row['j'], row['t']): row['x'] for index, row in dfx.iterrows() }
    z_load = {(row['j'], row['t']): row['z'] for index, row in dfz.iterrows() }
    phi_load = {(row['j'], row['t']): row['phi'] for index, row in dfphi.iterrows() }
    rho_load = {(row['t']): row['rho'] for index, row in dfrho.iterrows() }
    y_load = {(row['j'], row['t']): row['y'] for index, row in dfy.iterrows() }
    return x_load, z_load, y_load, phi_load, rho_load 

def toJson(x, z, filename):
    solout = {'solution':[]}
    solout['solution'].append( {'x': [ {'i':k[0], 'j':k[1], 'v':v} for k,v in x.items() ] } )
    solout['solution'].append( {'z' : [ {'i': k, 'v':v} for k,v in z.items()] } )
    with open (filename, 'w') as xout:
        json.dump(solout, xout, indent=4)
        xout.close()

def toExcel(vars : Vars, model, filename, filenamelog):
    path = 'results'
    dfpar = pd.DataFrame(columns=['parametro', 'valore'])
    dfx = pd.DataFrame(columns=['k', 'j', 't', 'x'])
    dfz = pd.DataFrame(columns=['j', 't', 'z'])
    dfphi = pd.DataFrame(columns=['j', 't', 'phi'])
    dfrho = pd.DataFrame(columns=['t', 'rho'])
    dfy = pd.DataFrame(columns=['j', 't', 'y'])
    
    writer = pd.ExcelWriter(path + '/' + filename, engine='xlsxwriter')
    for k,v in vars.x.items():
        dfx.loc[len(dfx)] = [k[0], k[1], k[2], v.x]
    dfx.to_excel(writer, sheet_name='x', index=False)

    for k,v in vars.z.items():
        dfz.loc[len(dfz)] = [k[0], k[1], v.x]
    dfz.to_excel(writer, sheet_name='z', index=False)

    for k,v in vars.phi.items():
        dfphi.loc[len(dfphi)] = [k[0], k[1], v.x]
    dfphi.to_excel(writer, sheet_name='phi', index=False)

    for k,v in vars.rho.items():
        dfrho.loc[len(dfrho)] = [k, v.x]
    dfrho.to_excel(writer, sheet_name='rho', index=False)

    for k,v in vars.y.items():
        dfy.loc[len(dfy)] = [k[0], k[1], v.x]
    dfy.to_excel(writer, sheet_name='y', index=False)

    dfpar.loc[len(dfpar)] = ['S', LS]
    dfpar.loc[len(dfpar)] = ['F', LF]
    dfpar.loc[len(dfpar)] = ['T', LT]
    dfpar.loc[len(dfpar)] = ['D', D]
    dfpar.loc[len(dfpar)] = ['B', B]
    dfpar.loc[len(dfpar)] = ['Sk', str(SK)]
    dfpar.loc[len(dfpar)] = ['Cj', str(CJ)]
    dfpar.loc[len(dfpar)] = ['phi_hat', phi_hat[1]]
    dfpar.loc[len(dfpar)] = ['phi_bar', phi_bar[1]]
    dfpar.loc[len(dfpar)] = ['rho_bar', rho_bar[1]]
    dfpar.loc[len(dfpar)] = ['alpha', alpha]
    dfpar.loc[len(dfpar)] = ['ObjVal', model.ObjVal]
    dfpar.loc[len(dfpar)] = ['Z1', vars.Z1.getValue()]
    dfpar.loc[len(dfpar)] = ['Z2', vars.Z2.getValue()]
    mipgap = 0.0 if (LINEAR and not BINARY) else model.MIPGap
    dfpar.loc[len(dfpar)] = ['MIPGap', mipgap]
    dfpar.loc[len(dfpar)] = ['RunTime', model.Runtime]
    dfpar.loc[len(dfpar)] = ['LINEAR', str(LINEAR)]
    dfpar.loc[len(dfpar)] = ['BINARY', str(BINARY)]
    dfpar.loc[len(dfpar)] = ['LOADSOLUTION', str(LOADSOLUTION)]
    dfpar.loc[len(dfpar)] = ['Z1UB', vars.Z1Ub]
    dfpar.loc[len(dfpar)] = ['Z2UB', vars.Z2Ub]
    dfpar.loc[len(dfpar)] = ['UB', vars.ub]

    dfpar.to_excel(writer, sheet_name='parametri', index=False)
    writer.save()

    df = pd.DataFrame(columns=['S', 'F', 'T', 'B', 'D', 'PhiHat', 'PhiBar', 'RhoBar', 'Eta', 'Z1', 'Z2', 
                          'ObjVal', 'time', 'GAP', 'LINEAR', 'BINARY', 'LOADSOL', 'MODELTYPE',
                          'Z1UB', 'Z2UB', 'UBHeur'])
    z1Val = vars.Z1.getValue()
    z2Val = vars.Z2.getValue()
    objval =  model.ObjVal
    runtime =  model.Runtime
    df.loc[0] = [LS, LF, LT, B, D, phi_hat_0, phi_bar_0, rho_bar_0, eta, z1Val, z2Val, 
                           objval, runtime, mipgap, LINEAR, BINARY, LOADSOLUTION, MODELTYPE, vars.Z1Ub, vars.Z2Ub, vars.ub ]

    if os.path.isfile(path + '/' + filenamelog):
            df.to_csv(path + '/' + filenamelog, mode='a', index=None, header=False)
    else:
        df.to_csv(path + '/' + filenamelog, index=None)

def getModelUB():
    model = gb.Model('ANOR')
    x = {(k,j, t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='x_{}_{}_{}'.format(k,j,t)) for (k,j) in A for t in T}
    z = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='z_{}_{}'.format(j,t)) for j in F for t in T}
    phi = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='phi_{}_{}'.format(j,t)) for j in F for t in T}
    rho = {t: model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='rho_{}'.format(t)) for t in T}
    y = {(j,t): model.addVar(vtype = gb.GRB.BINARY, name='y_{}_{}'.format(j,t)) for j in F for t in T0}

    Z1 = gb.quicksum( x[k,j,t]*phi[j,t] for (k,j) in A for t in T) 
    Z2 = gb.quicksum(rho[t] for t in T)
    Z = Z1 + Z2

    model.setObjective(Z, gb.GRB.MINIMIZE)

    model.addConstr( gb.quicksum( x[k,j,t] for (k,j) in A for t in T) == D, name='ctD')
    model.addConstrs( (gb.quicksum(x[k,j,t] for j in F ) <= SK[k,t] for k in S for t in T), name= 'ctSk' )
    model.addConstrs( (gb.quicksum(x[k,j, t] for k in S) <= CJ[j,t] for j in F for t in T), name='ctCj' )
    model.addConstr( gb.quicksum(z[j,t] for t in T for j in F) == B, name='ctZjB' )
    model.addConstrs( (B*y[j,t]  >= gb.quicksum(z[j,tau] for tau in range(1, t+1) ) for j in F for t in T), name='ctBy' )
    #model.addConstrs( (phi[j,t] *(1 + gb.quicksum(z[j,tau] for tau in range(1, t+1)) ) == y[j,t]*phi_hat[j]  
    #                + phi_bar[j]*(1-y[j,t])*(1 + gb.quicksum(z[j,tau] for tau in range(1, t+1)) ) for j in F for t in T ), name='ctPhi' )
    #model.addConstrs( (phi[j,t] >=  phi_bar[j]*(1-y[j,t]) for j in F for t in T ), name='ctPhiBis' )

    model.addConstrs( (phi[j,t] >= phi_hat[j]  + phi_bar[j]*(1-y[j,t]) for j in F for t in T ), name='ctPhi' )
    model.addConstrs( ( B * (gb.quicksum(z[j, tau] for tau in range(1, t+1))) >= y[j,t] for j in F for t in T), name='ctPhiBis' )
    

    model.addConstrs(  (rho[t] == gb.quicksum(rho_bar[t] * ((1 - alpha)**(tau - 1))*z[j,tau] for tau in range(1, t+1) for j in F) for t in T), name = 'ctRho' )

    model.addConstrs( (y[j,0] == 0 for j in F), name = 'y0')
    model.addConstrs((y[j,t] >= y[j,t-1] for t in T for j in F ), name = 'yt'  )
    #model.addConstrs((BigM*gb.quicksum(x[k,j,t] for k in S) >= z[j,t] for j in F for t in T), name = 'zx')
    if LB:
        model.addConstr( gb.quicksum( x[k,j,t]*phi[j,t] for (k,j) in A for t in T)  + gb.quicksum(rho[t] for t in T) >= LBValue, name='ctLB')

    # Load solution generated by lower bound model and set variables x, z, y to solution values
    if LOADSOLUTION:
        x_load, z_load, y_load, _, _ = loadSolExcel('results/' + flienameToLoad.format( "True", str(BINARY),
                                            "False", str(LS), str(LF), str(LT), str(B), str(D), str(phi_hat_0), 
                                            str(phi_bar_0), str(rho_bar_0), str(eta), "1"))
        #for k, v in x_load.items():
        #    x[k[0], k[1], k[2]].lb = v
        #    x[k[0], k[1], k[2]].ub = v

        for k, v in z_load.items():
            z[k[0], k[1]].lb = v
            z[k[0], k[1]].ub = v

        #for k, v in y_load.items():
        #    y[k[0], k[1]].lb = v
        #    y[k[0], k[1]].ub = v

    model.update()
    vars = Vars()
    vars.initVars(x, z, phi, rho, y, Z1, Z2)
    return vars, model

def heur(vars : Vars):
    # initialize to 0 all variables:
    vars.xStart   = {(k,j, t): 0.0 for (k,j) in A for t in T}
    vars.zStart   = {(j,t): 0.0 for j in F for t in T}
    vars.phiStart = {(j,t): 0.0 for j in F for t in T}
    vars.rhoStart = {t: 0.0 for t in T}
    vars.yStart   = {(j,t): 0 for j in F for t in T0}

    # iterate over x starting from the first period and start to load flow to the maximum counting storing the activated flows
    # untill all demand is satisfied
    flowAct = {}
    dResidual = D
    bResidual = B
    SKResidual = {k :SK[k] for k in SK.keys()}
    CJResidual = {k : CJ[k] for k in CJ.keys() }
    for t in T:
        for (k,j) in A:
            maxFlow = min(SKResidual[k, t], CJResidual[j,t])
            maxFlow = min(dResidual, maxFlow)
            if maxFlow == 0:
                continue
            vars.xStart[k,j,t] = maxFlow
            flowAct[(k,j,t)] = maxFlow
            dResidual -= maxFlow
            SKResidual[k,t] = SKResidual[k,t] - maxFlow
            CJResidual[j,t] = CJResidual[j,t] - maxFlow
            if dResidual == 0:
                break
        if dResidual == 0:
            break
    
    # for each activated link, activate investment z[j,t]
    nFlows = len(flowAct)
    bCurrent = B/nFlows
    for (k,j,t) in flowAct:
        vars.zStart[j,t] = vars.zStart[j,t] + bCurrent
        for tau in range (t,LT+1):
            vars.yStart[j,tau] = 1 # if j,t is 1, by construction of z, also y(j,t-1) will be 1

    # compute phi
    for j in F:
        for t in T:
            vars.phiStart[j,t] = vars.yStart[j,t] * ( phi_hat[j] /  ( sum(vars.zStart[j,tau] for tau in range (1, t+1)) + vars.yStart[j,t] - 1) ) 
            + phi_bar[j]*(1 - vars.yStart[j,t])

    #compute rho
    for t in T:
        vars.rhoStart[t] = sum( rho_bar[t] * vars.zStart[j, tau] * (1 - alpha)**(tau -t) for tau in range(t, LT+1) for j in F  )

    # compute UBValue:
    vars.Z1Ub = sum( vars.xStart[k,j,t] * vars.phiStart[j,t] for (k,j) in A for t in T)
    vars.Z1Ub = sum(vars.rhoStart[t] for t in T)
    vars.ub = vars.Z1Ub + vars.Z2Ub
    return


def getModel(UBMODEL = False):
    model = gb.Model('ANOR')
    vars = Vars()
    x = {(k,j, t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='x_{}_{}_{}'.format(k,j,t)) for (k,j) in A for t in T}
    z = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='z_{}_{}'.format(j,t)) for j in F for t in T}
    phi = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='phi_{}_{}'.format(j,t)) for j in F for t in T}
    rho = {t: model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='rho_{}'.format(t)) for t in T}
    y = {(j,t): model.addVar(vtype = gb.GRB.BINARY, name='y_{}_{}'.format(j,t)) for j in F for t in T0}

    Z1 = gb.quicksum( x[k,j,t]*phi[j,t] for (k,j) in A for t in T) 
    Z2 = gb.quicksum(rho[t] for t in T)
    Z = Z1 + Z2

    model.setObjective(Z, gb.GRB.MINIMIZE)

    model.addConstr( gb.quicksum( x[k,j,t] for (k,j) in A for t in T) == D, name='ctD') #(2)
    model.addConstrs( (gb.quicksum(x[k,j,t] for j in F ) <= SK[k,t] for k in S for t in T), name= 'ctSk' ) #(3)
    model.addConstrs( (gb.quicksum(x[k,j, t] for k in S) <= CJ[j,t] for j in F for t in T), name='ctCj' ) # (4)
    model.addConstr( gb.quicksum(z[j,t] for t in T for j in F) == B, name='ctZjB' ) # (5)
    model.addConstrs( (B*y[j,t]  >= gb.quicksum(z[j,tau] for tau in range(1, t+1) ) for j in F for t in T), name='ctBy' ) #(6)
    #model.addConstrs( (phi[j,t] *(1 + gb.quicksum(z[j,tau] for tau in range(1, t+1)) ) == y[j,t]*phi_hat[j]  
    #                + phi_bar[j]*(1-y[j,t])*(1 + gb.quicksum(z[j,tau] for tau in range(1, t+1)) ) for j in F for t in T ), name='ctPhi' )
    #model.addConstrs( (phi[j,t] >=  phi_bar[j]*(1-y[j,t]) for j in F for t in T ), name='ctPhiBis' )

    model.addConstrs( (phi[j,t] *( gb.quicksum(z[j,tau] for tau in range(1, t+1)) + y[j,t] -1) == y[j,t]*phi_hat[j]  
                    + phi_bar[j]*(1-y[j,t])*(gb.quicksum(z[j,tau] for tau in range(1, t+1)) + y[j,t] -1) 
                    for j in F for t in T ), name='ctPhi' ) # (7)
    #model.addConstrs( (phi[j,t] >= phi_hat[j]*y[j,t] +  phi_bar[j]*(1-y[j,t]) for j in F for t in T ), name='ctPhiBis' ) # (7bis)

    model.addConstrs( (gb.quicksum(x[k,j,tau] for k in S for tau in range(1, t+1))
                       >= (1 - y[j,t])*gb.quicksum(z[j,tau] for tau in range(1,t+1)) 
                       + y[j, t] * Lmin for j in F for t in T), name='ctXY' ) #(8)
    
    if UBMODEL:
        model.addConstrs( (phi[j,t] >= phi_hat[j]/B  + phi_bar[j]*(1-y[j,t]) for j in F for t in T ), name='ctPhi' )
        model.addConstrs( ( B * (gb.quicksum(z[j, tau] for tau in range(1, t+1))) >= y[j,t] for j in F for t in T), name='ctPhiBis' )
     

    model.addConstrs(  (rho[t] == 
                        gb.quicksum(rho_bar[t] * ((1 - alpha)**(tau - t))*z[j,tau] 
                                    for tau in range(t, LT+1) for j in F) for t in T), name = 'ctRho' ) #(9)
    model.addConstrs( (gb.quicksum(z[j,tau] for tau in range(1, t+1)) >= y[j,t]*Bmin for j in F for t in T ), name = 'ctZY') #(10)

    model.addConstrs( (y[j,0] == 0 for j in F), name = 'y0')
    model.addConstrs((y[j,t] >= y[j,t-1] for t in T for j in F ), name = 'yt'  )
    #model.addConstrs((BigM*gb.quicksum(x[k,j,t] for k in S) >= z[j,t] for j in F for t in T), name = 'zx')
    if LB:
        model.addConstr( gb.quicksum( x[k,j,t]*phi[j,t] for (k,j) in A for t in T)  + gb.quicksum(rho[t] for t in T) >= LBValue, name='ctLB')

    # Load solution generated by lower bound model and set variables x, z, y to solution values
    if LOADSOLUTION:
        x_load, z_load, y_load, _, _ = loadSolExcel('results/' + flienameToLoad.format( "True", str(BINARY),
                                            "False", str(LS), str(LF), str(LT), str(B), str(D), str(phi_hat_0), 
                                            str(phi_bar_0), str(rho_bar_0), str(eta), '1'))
        #for k, v in x_load.items():
        #    x[k[0], k[1], k[2]].lb = v
        #    x[k[0], k[1], k[2]].ub = v

        for k, v in z_load.items():
            z[k[0], k[1]].lb = v
            z[k[0], k[1]].ub = v

        #for k, v in y_load.items():
        #    y[k[0], k[1]].lb = v
        #    y[k[0], k[1]].ub = v

    model.update()
    vars.initVars(x, z, phi, rho, y, Z1, Z2)
    if (HEURISTIC_START):
        heur(vars)
        for key in x.keys():
            x[key].Start = vars.xStart[key]
        for key in y.keys():
            y[key].Start = vars.yStart[key]
        for key in z.keys():
            z[key].Start = vars.zStart[key]
        #for key in phi.keys():
        #    phi[key].Start = vars.phiStart[key]
        for key in rho.keys():
            rho[key].Start = vars.rhoStart[key]
        model.update()
    if (CHECK_HEUR_FEASIBILITY):
        heur(vars)
        model.addConstrs( (x[key] == vars.xStart[key] for key in x.keys()), name = 'xfeas')
        model.addConstrs( (y[key] == vars.yStart[key] for key in y.keys()), name = 'yfeas')
        model.addConstrs( (z[key] == vars.zStart[key] for key in z.keys()), name = 'xfeas')
        model.addConstrs( (rho[key] == vars.rhoStart[key] for key in rho.keys()), name = 'rhofeas')
        #model.addConstrs( (phi[key] == vars.phiStart[key] for key in phi.keys()), name = 'phifeas')


    return vars, model

def getModelLinear(binary = True):
    MM = 1000000
    pi = LS*LF*LT
    phi_tilde = max(max(phi_hat.values()),max(phi_bar.values()))
    model = gb.Model('ANOR')
    x = {(k,j, t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='x_{}_{}_{}'.format(k,j,t)) for (k,j) in A for t in T}
    z = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='z_{}_{}'.format(j,t)) for j in F for t in T}
    phi = {}
    rho = {}
    y = {}
    if binary:
        y = {(j,t): model.addVar(vtype = gb.GRB.BINARY, name='y_{}_{}'.format(j,t)) for j in F for t in T0}
    else:
        y = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, ub=1, name='y_{}_{}'.format(j,t)) for j in F for t in T0}


    Z1 = gb.quicksum( (1/B)*phi_hat[j]*x[k,j,t] + phi_bar[j]*x[k,j,t] - phi_tilde* (D/pi) *y[j,t] for (k,j) in A for t in T) 
    Z2 = gb.quicksum(rho_bar[t]*((1 - alpha)**(tau - t))*z[j,tau] for t in T for tau in range(t, LT+1) for j in F)
    Z = Z1 + Z2

    model.setObjective(Z, gb.GRB.MINIMIZE)

    model.addConstr( gb.quicksum( x[k,j,t] for (k,j) in A for t in T) == D, name='ctD')
    model.addConstrs( (gb.quicksum(x[k,j,t] for j in F ) <= SK[k,t] for k in S for t in T), name= 'ctSk' )
    model.addConstrs( (gb.quicksum(x[k,j, t] for k in S) <= CJ[j,t] for j in F for t in T), name='ctCj' )
    model.addConstr( gb.quicksum(z[j,t] for t in T for j in F) == B, name='ctZjB' )
    model.addConstrs( ( MM*gb.quicksum(z[j,tau] for tau in range(1,t+1)) >= y[j,t] for j in F for t in T  ), name='ctMMzy')
    model.addConstrs((y[j,t] >= y[j,t-1] for t in T for j in F ), name = 'yt'  )
    model.addConstrs( (y[j,0] == 0 for j in F), name = 'y0') 
    
    model.addConstrs((gb.quicksum(x[k,j,tau] for k in S for tau in range(1, t+1)) 
                     >= (1/B)*gb.quicksum(z[j,tau] for tau in range(1, t+1)) for j in F for t in T), name = 'xz')
    #model.addConstrs((BigM*gb.quicksum(x[k,j,t] for k in S) >= z[j,t] for j in F for t in T), name = 'zx')
    model.addConstrs( (gb.quicksum(z[j,tau] for tau in range(1, t+1)) >= y[j,t]*Bmin for j in F for t in T ), name = 'ctZY') #(10)


    model.update()
    vars = Vars()
    vars.initVars(x, z, phi, rho, y, Z1, Z2)
    return vars, model

def getLBNew(binary = True):
    MM = 1000000
    pi = LS*LF*LT
    phi_tilde = max(max(phi_hat.values()),max(phi_bar.values()))
    model = gb.Model('ANOR')
    x = {(k,j, t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='x_{}_{}_{}'.format(k,j,t)) for (k,j) in A for t in T}
    z = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='z_{}_{}'.format(j,t)) for j in F for t in T}
    phi = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, name='phi_{}_{}'.format(j,t)) for j in F for t in T}
    rho = {}
    y = {}
    if binary:
        y = {(j,t): model.addVar(vtype = gb.GRB.BINARY, name='y_{}_{}'.format(j,t)) for j in F for t in T0}
    else:
        y = {(j,t): model.addVar(vtype = gb.GRB.CONTINUOUS, lb=0, ub=1, name='y_{}_{}'.format(j,t)) for j in F for t in T0}


    Z1 = gb.quicksum( (1/B)*phi_hat[j]*x[k,j,t] + phi_bar[j]*x[k,j,t] - phi_tilde* (D/pi) *y[j,t] for (k,j) in A for t in T) 
    Z2 = gb.quicksum(rho_bar[t]*((1 - alpha)**(tau - t))*z[j,tau] for t in T for tau in range(t, LT+1) for j in F)
    Z = Z1 + Z2

    model.setObjective(Z, gb.GRB.MINIMIZE)

    model.addConstr( gb.quicksum( x[k,j,t] for (k,j) in A for t in T) == D, name='ctD')
    model.addConstrs( (gb.quicksum(x[k,j,t] for j in F ) <= SK[k,t] for k in S for t in T), name= 'ctSk' )
    model.addConstrs( (gb.quicksum(x[k,j, t] for k in S) <= CJ[j,t] for j in F for t in T), name='ctCj' )
    model.addConstr( gb.quicksum(z[j,t] for t in T for j in F) == B, name='ctZjB' )
    model.addConstrs( ( MM*gb.quicksum(z[j,tau] for tau in range(1,t+1)) >= y[j,t] for j in F for t in T  ), name='ctMMzy')

    model.addConstrs( (phi[j,t] <= phi_hat[j]*y[j,t] + phi_bar[j]*(1-y[j,t]) for j in F for t in T), name = 'ctlbphi1') 
    model.addConstrs( (phi[j,t] >= phi_hat[j]*y[j,t]/B + phi_bar[j]*(1-y[j,t]) for j in F for t in T), name = 'ctlbphi2') 

    model.addConstrs((gb.quicksum(x[k,j,tau] for k in S for tau in range(1, t+1)) 
                     >= (1/B)*gb.quicksum(z[j,tau] for tau in range(1, t+1)) for j in F for t in T), name = 'xz')

    model.addConstrs( (gb.quicksum(z[j,tau] for tau in range(1, t+1)) >= y[j,t]*Bmin for j in F for t in T ), name = 'ctZY') #(10)

    model.addConstrs((y[j,t] >= y[j,t-1] for t in T for j in F ), name = 'yt'  )
    model.addConstrs( (y[j,0] == 0 for j in F), name = 'y0') 
    
    #model.addConstrs((BigM*gb.quicksum(x[k,j,t] for k in S) >= z[j,t] for j in F for t in T), name = 'zx')

    model.update()
    vars = Vars()
    vars.initVars(x, z, phi, rho, y, Z1, Z2)
    return vars, model
  

if __name__=="__main__":
    TestN = 3
    TLIM = 1800
    LB = False
    LINEAR = False
    BINARY = True
    LOADSOLUTION = True
    HEURISTIC_START = True
    CHECK_HEUR_FEASIBILITY = False
    # 1 LB, 2 LB NEW, 3 NL, 4 NL LOAD LB, 5 UB, 6 Verify UB
    modelTypeString = 'NLMODEL'
    modeltypes = {"LB1" : 1, "LB2" : 2, "NLMODEL" : 3, "LOADLB" : 4, "UB" : 5, "VerifyUB" : 6, "HEURISTIC" : 7}
    MODELTYPE = modeltypes[modelTypeString]

    dfi = pd.read_excel('Instanceanor.xlsx')
    #dfi = pd.read_excel('InstanceValidation2.xlsx')
    
    if TestN > 0:
        #SSFF = [(31,31,0), (31,41,0), (41,41,0), (41, 51,1), (51,51,0)]
        #SSFF = [(6, 6, 0), (6,11,1), (11,11,3), (11,16,2), (16,16,2), (16,21,0),(21,21,0)]
        #SSFF = [(76,76,0), (76,101,0),(101,101,1)]
        #SSFF = [(6, 6, 0), (6,11,1), (11,11,3), (11,16,2), (16,16,2), (16,21,0),(21,21,0), (31,31,0), (31,41,0), (41,41,0), (41, 51,1), (51,51,0),(76,76,0), (76,101,0),(101,101,1)]
        #SSFF = [(6, 6, 11), (6,11,11), (11,11,11), (11,16,11), (16,16,11), (16,21,11),(21,21,11),(31,31,11), (31,41,11), (41,41,11), (41, 51,11), (51,51,11),(76,76,11), (76,101,11),(101,101,11)]
        SSFF = [(3,3,5,0)] # (LS, LF, LT)
        for index, row in dfi.iterrows():
            solved = int(row['solved'])
            #solved = 0 # fix in order to bypass check of solved solutions
            if solved == 1:
                print("instance already solved")
                continue
            LS = int(row['S'])
            LF = int(row['F'])
            LT = int(row['T'])
            seed = int(row['seed'])
            B = int(row['B'])
            Bmin = B/(LT*LF)
            D = int(row['D'])
            phi_hat_0 = int(row['PhiHat'])
            phi_bar_0 = int(row['PhiBar'])
            rho_bar_0 = float(row['RhoBar'])
            eta = float(row['Eta'])
            if not LINEAR:
                LBValue = float(row['LB'])
            S = [i for i in range(1,LS + 1)]
            F = [j for j in range(1,LF + 1)]
            T = [t for t in range(1, LT + 1)]
            T0 = [t for t in range(0,LT+1)] # Time horizon
            phi_hat = {j:phi_hat_0 for j in F }
            phi_bar = {j:phi_bar_0 for j in F}
            rho_bar = {k:rho_bar_0 for k in T}
            # if rhoDecrFactor != 1 rho_bar woul be not costant and must be computed -> rho_bar will be updated
            if rho_bar_0 != 1:
                rho_bar = getRhoBar(rho_bar_0, T)
            A= [(i,j) for (i,j) in itertools.product(S,F) ]
            np.random.seed(seed)  # per ripetibilità test
        

            SK =  { (k,t): v for (k,t),v in zip (itertools.product(S,T), np.random.choice(range(101,150), size=len(S)*len(T)) ) } # supplier capacity
            CJ =  { (k,t): v for (k,t),v in zip (itertools.product(F,T), np.random.choice(range(101,150), size=len(F)*len(T)) )}# plant capacity
            #SK =  { (k,t): v for (k,t),v in zip (itertools.product(S,T), np.random.choice(range(int(0.8*D/(len(S)*len(T))), int(1.8*D/(len(S)*len(T)))), size=len(S)*len(T)) ) } # supplier capacity
            #CJ =  { (k,t): v for (k,t),v in zip (itertools.product(F,T), np.random.choice(range(int(0.8*D/(len(S)*len(T))), int(1.8*D/(len(F)*len(T)))), size=len(F)*len(T)) )}

            if TestN == 1:
                SK =  { (k,t): v for (k,t),v in zip (itertools.product(S,T), [100 for i in range(len(S)*len(T))] ) } # supplier capacity
                CJ =  { (k,t): v for (k,t),v in zip (itertools.product(F,T), [100 for i in range(len(F)*len(T))] )}# plant capacity

            print("SK: ", SK)
            print("CJ: ", CJ)
            print("B ", B)
            print("Bmin: ", Bmin)
            print("D ", D)

            #if len(sys.argv) <= 1 or sys.argv[1].upper() == 'L':
            #    pass

            vars = None 
            model = None
            #model.write('modelQ.lp')
            # 1 LB, 2 LB NEW, 3 NL, 4 NL LOAD LB, 5 UB
            if MODELTYPE == 3 or MODELTYPE == 4:
                if MODELTYPE == 3:
                    LINEAR = False
                    LOADSOLUTION = False
                elif MODELTYPE == 4:
                    LINEAR = False
                    LOADSOLUTION = True
                vars, model = getModel()
                model.params.NonConvex = 2
            elif MODELTYPE == 1:
                LINEAR = True
                LOADSOLUTION = False
                vars, model = getModelLinear(binary=BINARY)
            elif MODELTYPE == 2:
                LINEAR = True
                LOADSOLUTION = False
                vars, model = getLBNew(binary=BINARY)
            elif MODELTYPE == 5:
                LINEAR = False
                LOADSOLUTION = False
                vars, model = getModelUB()
                model.params.NonConvex = 2
            elif MODELTYPE == 6:
                LINEAR = False
                LOADSOLUTION = False
                vars, model = getModel(UBMODEL=True)
                model.params.NonConvex = 2

            #model.write('model.lp')
            print("#### STARTING PROCESSING OF SOLUTION ###############################")
            print(filenameRes.format( str(LINEAR), str(BINARY),
                                                str(LOADSOLUTION), str(LS), str(LF), str(LT), str(B), str(D), str(phi_hat_0), 
                                                str(phi_bar_0), str(rho_bar_0), str(eta), str(MODELTYPE)))
            print("#####################################################################")

            model.params.TimeLimit = TLIM
            model.optimize()
            #ct = model.getConstrByName("ctRho[1]")
            #print(ct.getAttr("slack"))
            try:
                print("Z1: ", vars.Z1.getValue())
                print("Z2: ", vars.Z2.getValue())
                print("ObjVal: ", model.ObjVal)

                toExcel(vars, model, filenameRes.format( str(LINEAR), str(BINARY),
                                                str(LOADSOLUTION), str(LS), str(LF), str(LT), str(B), str(D), str(phi_hat_0), 
                                                str(phi_bar_0), str(rho_bar_0), str(eta), str(MODELTYPE)), 'results_log.csv')
            except Exception as e:
                #print(e.message)
                print("unable to solve instance because no upper bound found")
                pass
            """
            xtq = {k:v.x for k,v in x.items() }
            ztq =  {k:v.x for k,v in z.items() }
            print ('xt')
            print(xtq )
            print ('zt')
            print( ztq)
            toJson(xtq, ztq, 'xbestq_S{}_F{}_BD{}_s{}.json'.format(len(S), len(F), int(B/D), seed))
            optVal = model.ObjVal
            GAP = model.MIPGap
            with open('resultsQ.txt', 'a') as fo:
                fo.write(str(len(S)) + ',' + str(len(F)) + ',' + str(int(B/D)) + ',' + str(seed) 
                    + ',' + str(optVal) + ',' + str(GAP) + ',' + str(TLIM) + '\n')
                fo.close()
            """



