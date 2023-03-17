
from geneticalgorithm import geneticalgorithm
import numpy as np
import sys
import time
import matplotlib.pyplot as plt


class GA_parallel(geneticalgorithm):
    
    '''  Genetic Algorithm (Elitist version) for Python
    
    An implementation of elitist genetic algorithm for solving problems with
    continuous, integers, or mixed variables.
    
    
    
    Implementation and output:
        
        methods:
                run(): implements the genetic algorithm
                
        outputs:
                output_dict:  a dictionary including the best set of variables
            found and the value of the given function associated to it.
            {'variable': , 'function': }
            
                report: a list including the record of the progress of the
                algorithm over iterations
    '''
    #############################################################
    def __init__(self, function, dimension, variable_type='bool', \
                 variable_boundaries=None,\
                 variable_type_mixed=None, \
                 function_timeout=10,\
                 algorithm_parameters={'max_num_iteration': None,\
                                       'population_size':100,\
                                       'mutation_probability':0.1,\
                                       'elit_ratio': 0.01,\
                                       'crossover_probability': 0.5,\
                                       'parents_portion': 0.3,\
                                       'crossover_type':'uniform',\
                                       'max_iteration_without_improv':None},\
                     convergence_curve=True,\
                         progress_bar=True):

        super(GA_parallel, self).__init__(
            function = function,
            dimension = dimension,
            variable_type = variable_type,
            variable_boundaries= variable_boundaries,\
            variable_type_mixed=variable_type_mixed, \
            function_timeout=function_timeout,\
            algorithm_parameters=algorithm_parameters,\
            convergence_curve=convergence_curve,\
            progress_bar=progress_bar
            )

        ############################################################# 
    def run(self, init_pos):
        ############################################################# 
        self.best_history = []
        # Initial Population
        self.integers=np.where(self.var_type=='int')
        self.reals=np.where(self.var_type=='real')
        pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
        
        solo=np.zeros((self.pop_s, self.dim+1))
        var=np.zeros(self.dim)

        if np.any(init_pos == None):
            for p in range(0,self.pop_s):
            
                for i in self.integers[0]:
                    var[i]=np.random.randint(self.var_bound[i][0],\
                            self.var_bound[i][1]+1)  
                    solo[p, i]=var[i].copy()
                for i in self.reals[0]:
                    var[i]=self.var_bound[i][0]+np.random.random()*\
                    (self.var_bound[i][1]-self.var_bound[i][0])    
                    solo[p, i]=var[i].copy()
        else:
            solo[:, :self.dim] = init_pos


        obj=self.sim(solo[:, :self.dim])            
        solo[:, self.dim]=obj
        pop[:, :]=solo.copy()

        #############################################################

        #############################################################
        # Report
        self.report=[]
        self.test_obj=obj[-1]
        self.best_variable=var.copy()
        self.best_function=float('inf')
        ##############################################################   
                        
        t=1
        counter=0
        while t<=self.iterate:
            
            if self.progress_bar==True:
                self.progress(t,self.iterate,status="GA is running...")
            #############################################################
            #Sort
            pop = pop[pop[:,self.dim].argsort()]

    
            if pop[0,self.dim]<self.best_function:
                counter=0
                self.best_function=pop[0,self.dim].copy()
                self.best_variable=pop[0,: self.dim].copy()
            else:
                counter+=1
            

            self.best_history.append(self.best_function)
            #############################################################
            # Report

            self.report.append(pop[0,self.dim])
    
            ##############################################################         
            # Normalizing objective function 
            
            normobj=np.zeros(self.pop_s)
            
            minobj=pop[0,self.dim]
            if minobj<0:
                normobj=pop[:,self.dim]+abs(minobj)
                
            else:
                normobj=pop[:,self.dim].copy()
    
            maxnorm=np.amax(normobj)
            normobj=maxnorm-normobj+1

            #############################################################        
            # Calculate probability
            
            sum_normobj=np.sum(normobj)
            prob=np.zeros(self.pop_s)
            prob=normobj/sum_normobj
            cumprob=np.cumsum(prob)
  
            #############################################################        
            # Select parents
            par=np.array([np.zeros(self.dim+1)]*self.par_s)
            
            for k in range(0,self.num_elit):
                par[k]=pop[k].copy()
            for k in range(self.num_elit,self.par_s):
                index=np.searchsorted(cumprob,np.random.random())
                par[k]=pop[index].copy()
                
            ef_par_list=np.array([False]*self.par_s)
            par_count=0
            while par_count==0:
                for k in range(0,self.par_s):
                    if np.random.random()<=self.prob_cross:
                        ef_par_list[k]=True
                        par_count+=1
                 
            ef_par=par[ef_par_list].copy()
    
            #############################################################  
            #New generation
            pop=np.array([np.zeros(self.dim+1)]*self.pop_s)
            solo=np.zeros((self.pop_s, self.dim+1))

            for k in range(0,self.par_s):
                pop[k]=par[k].copy()
                
            for k in range(self.par_s, self.pop_s, 2):
                r1=np.random.randint(0,par_count)
                r2=np.random.randint(0,par_count)
                pvar1=ef_par[r1,: self.dim].copy()
                pvar2=ef_par[r2,: self.dim].copy()
                
                ch=self.cross(pvar1,pvar2,self.c_type)
                ch1=ch[0].copy()
                ch2=ch[1].copy()
                
                ch1=self.mut(ch1)
                ch2=self.mutmidle(ch2,pvar1,pvar2)               
                
                solo[k, :self.dim] = ch1.copy()
                solo[k+1, :self.dim] = ch2.copy()

            obj=self.sim(solo[:, :self.dim])            
            solo[:, self.dim]=obj
            pop[:, :]=solo.copy()

        #############################################################       
            t+=1
            if counter > self.mniwi:
                pop = pop[pop[:,self.dim].argsort()]
                if pop[0,self.dim]>=self.best_function:
                    t=self.iterate
                    if self.progress_bar==True:
                        self.progress(t,self.iterate,status="GA is running...")
                    # time.sleep(2)
                    t+=1
                    self.stop_mniwi=True
                
                self.best_history.append(self.best_function)
                
        #############################################################
        #Sort
        pop = pop[pop[:,self.dim].argsort()]
        
        if pop[0,self.dim]<self.best_function:
                
            self.best_function=pop[0,self.dim].copy()
            self.best_variable=pop[0,: self.dim].copy()
        #############################################################
        # Report
        self.best_history.append(self.best_function)

        self.report.append(pop[0,self.dim])
        
        
        self.output_dict={'variable': self.best_variable, 'function':\
                          self.best_function}
        if self.progress_bar==True:
            show=' '*100
            sys.stdout.write('\r%s' % (show))
        sys.stdout.write('\r The best solution found:\n %s' % (self.best_variable))
        sys.stdout.write('\n\n Objective function:\n %s\n' % (self.best_function))
        sys.stdout.flush() 
        re=np.array(self.report)
        if self.convergence_curve==True:
            plt.plot(re)
            plt.xlabel('Iteration')
            plt.ylabel('Objective function')
            plt.title('Genetic Algorithm')
            plt.show()
        
        if self.stop_mniwi==True:
            sys.stdout.write('\nWarning: GA is terminated due to the'+\
                             ' maximum number of iterations without improvement was met!')

###############################################################################     
    def evaluate(self):
        return self.f(self.temp)
###############################################################################    
    def sim(self,X):
        self.temp=X.copy()
        obj = self.evaluate()
        return obj