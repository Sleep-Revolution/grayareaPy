
from src.MixtureModels import *
import pickle
import os


class RunExp:
    def __init__(self,list_pi=[""],list_theta_i_m=[""],list_N=[""],list_E=[""],list_Distr=[""],list_alpha=[""]):
        
        dict_exp = []
        
        for n in list_N:
            for e in list_E:
                for p in list_pi:
                    for dist in list_Distr:
                        for theta in list_theta_i_m:
                            for alpha in list_alpha:
                                temp = {"Results":[]}
                                
                                temp["pi"] = p
                                temp["theta_i_m"] = theta
                                temp["N"] = n
                                temp["E"] = e
                                temp["Distribution"] = dist
                                temp["Alpha"] = alpha
                                
                                dict_exp.append(temp)
        self.dict_exp = dict_exp

    def Run(self,nRep = 30):
        np.random.seed(12)
        seed_array = np.random.randint(0,2**16,nRep)
        
        f = IntProgress(min=0, max=len(self.dict_exp),description="nExp= 0"+"/"+str(len(self.dict_exp)))
        display(f)
        
        for exp in range(len(self.dict_exp)):
            f.description = "nExp= "+str(exp+1)+"/"+str(len(self.dict_exp))
            if exp == 0:
                h = IntProgress(min=0, max=nRep,description="nRep= 0"+"/"+str(nRep))
                display(h)
            else:
                h.value = 0
            for n in range(nRep):
                h.description = "nRep= "+str(n+1)+"/"+str(nRep)
                results = {}
                GenSample = GenMixtSampleFromCatEns(E=self.dict_exp[exp]["E"],pi_Z=self.dict_exp[exp]["theta_i_m"])
                Y,Z = GenSample.generate(N=self.dict_exp[exp]["N"],
                                         pi=self.dict_exp[exp]["pi"],
                                         seed=seed_array[n],
                                         distribution="Categorial")
                X_,Z_X = GenSample.generate(N=self.dict_exp[exp]["N"],
                                            Z=Z,
                                            seed=seed_array[n],
                                            distribution=self.dict_exp[exp]["Distribution"])
                MM = MixtModel(E=self.dict_exp[exp]["E"],distribution=self.dict_exp[exp]["Distribution"])
                
                MM.fit(X_,method="K-means",init="k-means++")
                results["M"] = len(MM.model["theta_i"]["pi"])
                
                results["model_init"] = MM.model_init
                results["criteria"] = MM.distribution.criteria

                MM = MixtModel(E=self.dict_exp[exp]["E"],distribution=self.dict_exp[exp]["Distribution"])
                MM.fit(X_,method="K-means",init="k-means++",M=len(self.dict_exp[exp]["pi"]))
                self.MM = MM
                results["model"] = MM.model
                results["threshold"] = MM.distribution.threshold
                results["dU2"] = MM.distribution.dU2max
                results["dU2"] = MM.distribution.dU2max
                
                CM,RMSE = self.Evaluation(X_,
                                          MM.model,
                                          trueZ=Z,
                                          distribution=self.dict_exp[exp]["Distribution"],
                                          pi=self.dict_exp[exp]["pi"],
                                          theta_i_m=self.dict_exp[exp]["theta_i_m"])
                
                results["RMSE"] = RMSE
                results["CM"] = CM
                
                if (self.dict_exp[exp]["Distribution"] == "Dirichlet") & (not (self.dict_exp[exp]["Alpha"]=="")):
                    # Dirichlet sample generated from alpha parameter
                    X_,Z_D = GenSample.generate(N=self.dict_exp[exp]["N"],
                                                    alpha=self.dict_exp[exp]["Alpha"],
                                                    Z=Z,
                                                    seed=seed_array[n],
                                                    distribution="Dirichlet")
                    MM = MixtModel(E=self.dict_exp[exp]["E"],distribution=self.dict_exp[exp]["Distribution"])
                    model, criteria = MM.distribution.searchM(X_,method="K-means",init="k-means++")
                    results["model_init_alpha"] = model
                    results["criteria_alpha"] = criteria
                    results["M_alpha"]  = len(model["theta_i"]["pi"])
                    MM.fit(X_,method="K-means",M=len(self.dict_exp[exp]["pi"]),init="k-means++")
                    CM,RMSE = self.Evaluation(X_,MM.model,
                                              trueZ=Z,
                                              distribution=self.dict_exp[exp]["Distribution"],
                                              pi=self.dict_exp[exp]["pi"],
                                              theta_i_m=self.dict_exp[exp]["theta_i_m"],
                                              alpha=self.dict_exp[exp]["Alpha"])
                    results["model_alpha"] = MM.model
                    results["RMSE_alpha"] = RMSE
                    results["CM_alpha"] = CM
                h.value +=1
                        
                self.dict_exp[exp]["Results"].append(results)
            f.value +=1
        self.dict_exp.append(seed_array)

    
    def Evaluation(self,X,model,trueZ,distribution,pi,theta_i_m,alpha=None):
        clusters = self.MM.distribution.predict(X,supracluster=False)
        CM = confusion_matrix(trueZ,clusters)
        
        if (distribution == "Dirichlet") & (not (alpha is None)):
            RMSE = sum(((alpha-model["theta_i"]["theta_i_m"])**2).sum(axis=1))/(alpha.shape[0]*alpha.shape[1])
            RMSE += sum(((pi-model["theta_i"]["pi"])**2))/(alpha.shape[0])
        else:
            if (distribution == "Dirichlet"):
                RMSE = 0
                for m in set(trueZ):
                    RMSE += sum((X[m==trueZ].mean(axis=0)-theta_i_m[m,:]/sum(theta_i_m[m,:]))**2)/theta_i_m.shape[1]
                RMSE = RMSE/len(set(trueZ))
                RMSE += sum(((pi-model["theta_i"]["pi"])**2))/(theta_i_m.shape[0])
                
            else:
                RMSE = sum(((theta_i_m-model["theta_i"]["theta_i_m"])**2).sum(axis=1))/(theta_i_m.shape[0]*theta_i_m.shape[1])
                RMSE += sum(((pi-model["theta_i"]["pi"])**2))/(theta_i_m.shape[0])
        return CM, np.sqrt(RMSE)
    
    def saveEXP(self,name):
        if os.path.isdir(os.path.join('save_data')):
            path_save = os.path.join('save_data')
        else:
            path_save = os.mkdir('save_data')
            path_save = os.path.join('save_data')
        
        path_name = os.path.join(path_save,name+'.pickle')
        i=0
        while os.path.exists(path_name):
            path_name = os.path.join(path_save,name+str(i)+'.pickle')
            i+=1
        with open(path_name, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    def loadEXPPickle(self,path_name):
        with open(path_name, 'rb') as file:
            runEXP = pickle.load(file)
        return runEXP
