#GLADIATOR.py

import sys
import numpy as np
import random
import math
#import os
import pandas as pd
import scipy as sp
import networkx as nx
from sklearn.metrics.pairwise import pairwise_distances
from distutils.version import LooseVersion

#Reading the PPI from a file formated Protein1\tProtein2\tNotes
def ReadInteractome(FileName, SKIP):
	column_names = ('P1', 'P2', 'data_source')
	#reading p1 and p2 as strings
	PPIs = pd.read_table(FileName, header=None, skiprows=SKIP, names=column_names, sep='\t', converters = {'P1':str}, dtype={'P2':str})
	#removing all self loops (2869 interactions)
	PPIs = PPIs[PPIs['P1'] != PPIs['P2']]
	AllProts = pd.concat((PPIs['P1'], PPIs['P2'])).unique()
	return PPIs, AllProts

#Loading KnownDisPS used by GLADIATOR to calculate the SeedPS
def ReadBarabasiDiseaseProt(FileName):
	#file format:  disease number_of_all_genes	number_of_OMIM_genes number_of_GWAS_genes OMIM_genes(;seperated) GWAS_genes(;seperated)
	DiseaseProts = {}
	f = open(FileName, 'r')
	for l in f:
		if l[0]!='#': #header
			fields = l.strip().split('\t')
			DiseaseProts[fields[0]] = {}
			DiseaseProts[fields[0]]['OmimProts']=[]
			DiseaseProts[fields[0]]['GwasProts']=[]
			if fields[2] != '0':
				DiseaseProts[fields[0]]['OmimProts']=fields[4].split(';')
				if fields[3] != '0':
					DiseaseProts[fields[0]]['GwasProts']=fields[5].split(';')
			else:
				DiseaseProts[fields[0]]['GwasProts']=fields[4].split(';')
	f.close()
	return DiseaseProts

def ReplaceSeed(DiseaseProts, SeedPSfName):
	f = open(SeedPSfName, 'r')
	for l in f:
	    DName,SeedPS = l.strip().split('\t')
	    if DName in DiseaseProts.keys():
	        DiseaseProts[DName] = SeedPS.split(',')
	f.close()
	return DiseaseProts


##Class Module contains the current module for each disease, and can operate on this module (remove or add random protein etc)##
class Module():
    def __init__(self, Name, Ind, DiseaseProt, PPIGraph):
        self.Name = Name #disease name
        self.Ind = Ind   #index in similarity matrix
        self.DiseaseProt = list(DiseaseProt) #a copy of all know disease proteins (KnownDisPS).
        self.ProtsToRem = []  #proteins in module excluding seed (ModulePS\SeedPS)
        #1. searching for biggest connected component as an initial seed (SeedPS)
        SubG = PPIGraph.subgraph(DiseaseProt)
        if LooseVersion(nx.__version__) >= LooseVersion("1.9"):
            self.InModule = list(sorted(list(nx.connected_components(SubG)), key = len, reverse=True)[0]) #0 return the biggest connected component out of all cc found
        else:
            self.InModule = nx.connected_components(SubG)[0]
        self.SeedProts = list(self.InModule) #the list will copy the list instead of referencing to it.

        #2. adding immidiate neighbors of these proteins to ProtToAdd list
        protNei = np.unique(sum([PPIGraph.neighbors(prot) for prot in self.InModule],[]))#all neighbors of all disease prot in cc
        self.ProtsToAdd  = [p for p in protNei if p not in self.InModule] #only prots which are not already in the module
        #3. creating a subgraph of all potential proteins in the module which include the cc proteins, thier neighbors, disease proteins which are the neighbor of the neighbor
        #i.e in distance 2 from the original start disease prot, and their neighbors. to decrease search space and computational time
        protNeiNei = np.unique(sum([PPIGraph.neighbors(prot) for prot in self.ProtsToAdd],[]))#all neighbors of all disease prot in cc
        PotentialDiseaseProt = [p for p in DiseaseProt if (p in protNeiNei) and (p not in self.InModule)]
        protNeiDNei = np.unique(sum([PPIGraph.neighbors(prot) for prot in PotentialDiseaseProt],[]))#all neighbors of all disease prot in cc
        #protNeiDNei = [p for p in protNeiDNei if p not in PotentialDiseaseProt and p not in self.InModule and p not in self.ProtsToAdd] #no need as it is only used to create the subgraph
        self.SubGraph= PPIGraph.subgraph(list(self.InModule) + list(self.ProtsToAdd) + list(PotentialDiseaseProt) + list(protNeiDNei))
        #Calc density
        sg = self.SubGraph.subgraph(self.InModule)
        self.BU()

    #adding a random protein to the module for the set of current module neighbors
    def AddProt(self):
        if len(self.ProtsToAdd) < 1:
            return []

        self.BU()
        ProtToAdd = self.ProtsToAdd[random.randint(0,len(self.ProtsToAdd)-1)]

        #removing from add list and adding to removal list, and updating module list
        self.ProtsToAdd.remove(ProtToAdd)
        self.ProtsToRem.append(ProtToAdd)
        self.InModule.append(ProtToAdd)
        #update density
        #update ProtsToAdd list with neighbors of that protein which now can also be added to the module
        self.ProtsToAdd = list(np.unique(list(self.ProtsToAdd) + [p for p in self.SubGraph.neighbors(ProtToAdd) if p not in self.InModule]))
        return [ProtToAdd]

    #removing a random protein fom current module and subsequently all proteins disconnected from seed
    def RemProt(self):
        if len(self.ProtsToRem) < 1:
            return []

        self.BU()
        ProtToRem = self.ProtsToRem[random.randint(0,len(self.ProtsToRem)-1)]
        #recalc the module so that we have connected component.
        self.InModule.remove(ProtToRem)
        sg = self.SubGraph.subgraph(self.InModule)
        #test which CC contains the seed.
        if LooseVersion(nx.__version__) >= LooseVersion("1.9"):
            for CC in sorted(list(nx.connected_component_subgraphs(sg)), key = len, reverse=True):
                if self.SeedProts[0] in CC:
                    self.InModule = list(CC)
                    break
        else:
            for CC in nx.connected_components(sg):
                if self.SeedProts[0] in CC:
                    self.InModule = list(CC)
                    break
        #update ProtToRem
        self.ProtsToRem = [p for p in self.InModule if p not in self.SeedProts] #possible proteins to remove from module in next step

        #update ProtsToAdd list - removing proteins which are neighbors of this protein and not neighbors of any other protein in the module.
        ModuleNei = np.unique(sum([self.SubGraph.neighbors(prot) for prot in self.InModule],[]))#all neighbors of all disease prot in module
        self.ProtsToAdd  = [p for p in ModuleNei if p not in self.InModule] #only prots which are not already in the module

        return [p for p in self.InModuleBU if p not in self.InModule] #proteins which needs to be removed from module

    #undo last protein addition/removal if rejected by the annealing
    def Revert(self):
        self.ProtsToAdd = list(self.ProtsToAddBU)
        self.ProtsToRem = list(self.ProtsToRemBU)
        self.InModule = list(self.InModuleBU)

    #Bachup current state
    def BU(self):
        self.ProtsToAddBU = list(self.ProtsToAdd)
        self.ProtsToRemBU = list(self.ProtsToRem)
        self.InModuleBU = list(self.InModule)



#this class implement the algorithm behind the annealing - modules calculation, similarity calculation, and so on
class FindModules():
    #init function gets a dictionary of disease name and protein vector name (DiseaseProt). PPI association (P1,P2). the correlation to compare to (disease*disease mat) with corresponding disease name
    def __init__(self, DiseaseProt: object, PPI: object, DiseaseNames: object, PhenCorr: object) -> object:
        self.PPIGraph, self.ProtNames = self.BuildPPIGraph(PPI) #returns protein name list and ppi graph from the protein index
        self.PhenCorr = PhenCorr # constant correlation vector (e.g. phenotypic) sized diseases * diseases
        self.DiseaseNames = list(DiseaseNames) # given in the order of the PhenCorr matrix
        #state => diseases * protein matrix, holding the current disease moudle;
        self.state, self.Modules = self.CalcFirstModuleProt(DiseaseProt) #state = current module, ProtInd = protein name to index. DiseaseInd = diseae name to index. ProtName = in the order of the matrix
        self.ModuleCorr  = 1-pairwise_distances(self.state, metric='jaccard') #correlation vector (disease protein jaccard) to update every move, size of diseases * diseases
        self.triu_ind = np.triu_indices(self.ModuleCorr.shape[0],1) #relevant index to compare


    #one step of the annealing, choosing a random disease to change, and a random action to perform (add or remove protein)
    def move(self):
        """add or remove protein to the solution"""
        #randomly select a module\
        module2change = random.randint(0, self.state.shape[0] - 1)
        #self.StateToRevert = self.state[module2change,:].copy()
        RAS = random.randint(0,1)
        if RAS:
            ProtsI = self.Modules[module2change].AddProt()
        else:
            ProtsI = self.Modules[module2change].RemProt()

        #change module matrix accordingly
        for ProtI in ProtsI:
            self.state[module2change,ProtI] = int( not( bool(self.state[module2change,ProtI]))) # 0=>1 and 1=>0
        if len(ProtsI)>0: #len(ProtsI)==0 if e.g., no protein was available for removal
            self.UpdateCorr(module2change)
        return module2change, ProtsI,RAS

    #undo last protein addition/removal if rejected by the annealing
    def Revert(self,ModuleI, ProtsI):
         for ProtI in ProtsI:
            self.state[ModuleI,ProtI] = int ( not ( bool (self.state[ModuleI,ProtI]))) # 0=>1 and 1=>0
         self.Modules[ModuleI].Revert()
         self.UpdateCorr(ModuleI)

    #calculate the current energy function
    def energy(self):ww
        return sp.spatial.distance.sqeuclidean(self.ModuleCorr[self.triu_ind],self.PhenCorr[self.triu_ind])
        #return pairwise_distances(self.ModuleCorr[self.triu_ind],self.PhenCorr[self.triu_ind], metric='euclidean')[0][0]

    # update the jaccard distance between the module that was changed and all its neighbors
    def UpdateCorr(self,ModuleInd):
        x = np.zeros((1,13397)) # 13397 number of protein. should change to state.shape[2]
        x[0,:] = self.state[ModuleInd,:]
        self.ModuleCorr[ModuleInd,:] = 1 - pairwise_distances(x,self.state, metric = 'jaccard')
        self.ModuleCorr[:,ModuleInd] = self.ModuleCorr[ModuleInd,:]


    #Obtain the initial modules, i.e., SeedPS for all diseases
    def CalcFirstModuleProt(self,DiseasesProts):
        AllModuleProt = np.zeros((len(self.DiseaseNames),len(self.ProtNames)))
        Modules = []
        for i in range(len(self.DiseaseNames)):
            DiseaseProt = [self.ProtNames.index(p) for p in DiseasesProts[self.DiseaseNames[i]]]
            Modules.append(Module(self.DiseaseNames[i], i, DiseaseProt , self.PPIGraph))
            AllModuleProt[i, list(Modules[-1].InModule)]=1 #all protein in the biggest cc found for the disease
        return AllModuleProt, Modules


    #building the graph by protein index in the module and not by protein name
    def BuildPPIGraph(self,PPIs):
        ProtNames = list(np.unique(np.union1d(PPIs.P1.unique(), PPIs.P2.unique())))
        PPIGraph = nx.Graph()
        for ppi in PPIs.iterrows():
            PPIGraph.add_edge(ProtNames.index(ppi[1]['P1']),ProtNames.index(ppi[1]['P2']),weight=1)
        return PPIGraph, ProtNames


    def getProt(self):
        return self.ProtNames

## end class


#The annealing procedure
def Annealing(ModuleFinder,TMax,TMin, Alpha, iInter):
    ProgPer = 200 #progress bar to display
    NumOfSteps=int(math.log((TMin/TMax), Alpha)*iInter)
    ReportToBar = range(int(NumOfSteps/ProgPer),NumOfSteps,int(NumOfSteps/ProgPer)) #step index to report, assuming that we display the progress ProgPer times
    sys.stdout.write("[%s]" % (" " * ProgPer))
    sys.stdout.flush()
    sys.stdout.write("\b" * (ProgPer+1)) # return to start of line, after '['
    OldE = ModuleFinder.energy() #curr energy to compare to after each move
    T = TMax
    i = 1
    while T > TMin:
        ModuleI,ProtsI,RAS = ModuleFinder.move() #changes self.state to its neighbor
        if len(ProtsI)==0: #no move was done - no need to do anything or even count it as a move
            continue
        NewE = ModuleFinder.energy()
        deltaE = (OldE-NewE)
        if deltaE > 0:
            ap = 5
        else:
            try:
                ap = math.exp((deltaE)/T)
            except:
                print('ap failed with params:',deltaE,T)
                return ModuleFinder.state, OldE
        if ap > random.random(): #random returns uniform distribution in range [0,1]
            OldE = NewE
        else:
            #revert back to old state
            ModuleFinder.Revert(ModuleI, ProtsI)
        i += 1
        if i % iInter == 0:
            T = T*Alpha
        if i in ReportToBar:
            sys.stdout.write("*")
            sys.stdout.flush()

    sys.stdout.write("\n")
    return ModuleFinder.state, OldE



usage = """USAGE: GLADIATOR.py  
   -o  --OutFileName  [file_name]    Output file, store predicted module. [default: GLADIATOR_Modules.txt]
   -n  --NetworkfName [file_name]    Input PPI network file. File format: Protein1 \t Protein2 \t Comments [default: Interactome.tsv]
   -s  --SeedPSfName  [file_name]    Initial SeedPS file. Connected component in the PPI served as Seed for diseases. [default: KnownDisPS.tsv]
   -p  --PhenSimMat   [file_name]    Phenotyic similarity file. Similarity score for each disease pairs for the objective function  [default: PhenSimMat.zip]
   
   -h, --help                 print this help 
"""

if __name__ == '__main__':
    import sys
    import getopt

    PPIFName="Interactome.tsv";MyPPI = True; RandSeed = "Diseases"; OutFName = 'GLADIATOR_Modules.txt';SeedPSfName=''
    TMax = 5; TMin = 1e-25; TempAlpha = 0.995; StepsInTemp = 200
    DiseaseGeneFName = 'KnownDisPS.tsv'; PhenFName = 'PhenSimMat.tsv';
    try:
        opts, args = getopt.getopt(sys.argv[1:], "s:o:n:p:h", ["SeedPSfName=", "OutFileName=","NetworkfName=","PhenSimMat=","help"])
    except getopt.GetoptError:
        print(usage)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(usage)
            sys.exit()
        if opt in ("-n", "--NetworkfName"):
            PPIFName = arg
            MyPPI=False
        elif opt in ("-o", "--OutFile"):
            OutFName = arg
        elif opt in ("-s", "--SeedPSfName"):
            SeedPSfName = arg
        elif opt in ("-p", "--PhenSimMat"):
            PhenFName = arg

    random.seed(RandSeed)

    if MyPPI:
        PPIs, AllProts = ReadInteractome(PPIFName,26)
        PPIs.drop('data_source', axis=1, inplace=True)
    #loading relevant disease proteins
    else:
        #load other PPI data source in format P1 tab P2
        PPIs, AllProts = ReadInteractome(PPIFName,0)

    #loading relevant disease proteins
    DiseaseProts = ReadBarabasiDiseaseProt(DiseaseGeneFName)

    #load phenotype similarity matrix
    PhenSimDF = pd.read_csv(PhenFName,sep='\t',index_col=0)
    PhenSimMat = PhenSimDF.as_matrix()
    PhenDiseaseName = PhenSimDF.index.values

    #building the KnownDisPS dictionary
    DiseasesProtDict = {}
    for d in PhenDiseaseName:
        DProt = DiseaseProts[d]['GwasProts']+DiseaseProts[d]['OmimProts']
        DiseasesProtDict[d]= [p for p in DProt if p in AllProts]

    #if the user provide alternative seed for diseases replace the given KnwonDisPS with the one given in this file
    if len(SeedPSfName)>0:
        DiseaseProts = ReplaceSeed(DiseasesProtDict, SeedPSfName)


    ModuleFinder = FindModules(DiseasesProtDict,PPIs,PhenDiseaseName,PhenSimMat)

    Modules, Corr = Annealing(ModuleFinder,TMax,TMin, TempAlpha, StepsInTemp)
    ProtNames = ModuleFinder.getProt()
    f = open(OutFName,'w')
    f.write('Disease names\tModules predicted by GLADIATOR (entrez IDs)\n')
    for i in range(len(PhenDiseaseName)):
        f.write(PhenDiseaseName[i] + '\t')
        prots = np.where(Modules[i,:])[0]
        f.write(', '.join(np.array(ProtNames)[prots]))
        f.write('\n')
    f.close()

