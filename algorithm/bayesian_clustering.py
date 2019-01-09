import numpy as np
from scipy.special import factorial,multigammaln
from decimal import Decimal

## The node class which has a tree-shaped structure for the hierarchical clustering.
## Each node acts as a cluster and initially starts with one point, when we start combining the nodes 
## we update the left, right and points sets accordingly. Furthermore each nodes holds the probability
## of beloning together with every other nodes, these are also updated accordingly. Using dictionaries
## allow us to do faster calculations compared to numpy arrays.


class Node:

    def __init__(self,p,alpha,i):
        
        self.single=True
        self.points=set()
        self.points.add(p)
        self.d=alpha
        self.number=i
        self.left=None
        self.right=None
        self.ph=0
## Add list that holds probability with other nodes
        self.pit={}
        self.rit={}

## Function for adding new points
    def add(self,x):
        self.points.add(x)
        
    def add_all(self,x):
        self.points=x
        
    def remove(self,x):
        self.points.remove(x)
        
 ## Function for combining nodes together into one.       
    def combine(self,y,alpha,i=None):

        p=self.points.union(y.points)
        z=Node(1,self.d,self.number)
        if i!=None:
            z.number=i
        z.left=self
        z.right=y
        z.remove(1)
        z.points=p
        z.d= alpha*factorial(len(p)-1)+self.d*y.d
        z.single=False
        return z



## Function that calculates the probability of points in a node belonging together
## Use of decimals is crucial because some probabilites are rather small and we end up with overflows.
        
def prob_hypo(X,kappa0,v0,mu0,eta0):
    nf,df= X.shape
    n=Decimal(nf)
    d=Decimal(df)
    a= (1/(Decimal(np.pi))**(n*d/2))
    b=Decimal(multigammaln((v0+nf)/2,df)/multigammaln(v0/2,df))
    S=np.zeros((df,df))
    for i in range(nf):
        o=X[i]-X.mean(axis=0)
        S+=np.outer(o,o)
    etan=(eta0) + S + (kappa0*nf/(kappa0+nf))*np.outer((X.mean(axis=0)-mu0),(X.mean(axis=0)-mu0))
    c=Decimal(np.linalg.det(eta0)**(v0/2))/(Decimal(np.linalg.det(etan))**((Decimal(v0)+n)/2))
    d=Decimal(kappa0/(kappa0+nf))**(d/2)
    return float(a*b*c*d)


## Posterior and Probability calculations between 2 nodes:
## @pit is the probability that the data int two nodes belonging together and 
## @rit is the posterier probability calculation.

def get_pi_ri(i,j,alpha):
    clust_k=i.combine(j,alpha)
    nk=len(clust_k.points)
    dk=clust_k.d
    pi=alpha*factorial(nk-1)/dk 
    all_points=list(clust_k.points)
    ph=prob_hypo(X[all_points][:],kappa0,v0,mu0,eta0) 
    pit=ph*pi+ (1-pi)*i.ph*j.ph
    rit=(pi*ph)/pit
    return pit,rit
    
## Function that calculates the maximum value in a dictionary and returns the key-value combination
def get_dict_max(d):
    ind=0
    m=0
    for i in d:
        if d[i]>=m:
            m=d[i]
            ind=i
    return ind,m

## Function that retursn the node with the specific number from a list of nodes/
def get_node(i,nodes):
    for node in nodes:
        if node.number==i:
            return node

## Function that initiates rit and pit calculations by going over each node combination.
## Further updates only require to update according to new merges which saves time    
## Returns a list of nodes where with each point belonging to one node.
## alpha,kappa0,v0,mu0,eta0 are prior paremeters.
def init(X,alpha,kappa0,v0,mu0,eta0):
    x=[]
    for i in range(len(X)):
        node=Node(i,alpha,i)
        node.ph=prob_hypo(X[[i]],kappa0,v0,mu0,eta0)
        x.append(node)
    
    n=len(x)
    for i in range(n):
        for j in range(i+1,n):
            p,r=get_pi_ri(x[i],x[j],alpha)
            x[i].pit[j]=p
            x[i].rit[j]=r  
        
    return x

## Function that creates a new node from the combination of 2 nodes by updating the left , right and ph values in the new node
## Later we drop the nodes used for the merge from our list. 
def change_nodes(i,j,n,nodes,alpha):
 
    n1=get_node(i,nodes)
    n2=get_node(j,nodes)

    new_node=n1.combine(n2,alpha,i=n)
    new_node.ph=get_pi_ri(n1,n2,alpha=alpha)[0] 
    nodes.remove(n1)
    nodes.remove(n2)
    for node in nodes: ## Loop through the remaining nodes to update the dictionary of the newly created node
        
        new_node.pit[node.number],new_node.rit[node.number]=get_pi_ri(node,new_node,alpha=alpha)   
        if n1.number in node.rit:
            del node.rit[n1.number]
        if  n2.number in node.rit:
            del node.rit[n2.number]
    nodes.append(new_node)
    return nodes 


## Function that calculates the probability of points being in the same cluster
        
def prob_hypo(X,kappa0,v0,mu0,eta0):
    from decimal import Decimal
    nf,df= X.shape
    n=Decimal(nf)
    d=Decimal(df)
    a= (1/(Decimal(np.pi))**(n*d/2))
    b=Decimal(multigammaln((v0+nf)/2,df)/multigammaln(v0/2,df))
    S=np.zeros((df,df))
    for i in range(nf):
        o=X[i]-X.mean(axis=0)
        S+=np.outer(o,o)
    etan=(eta0) + S + (kappa0*nf/(kappa0+nf))*np.outer((X.mean(axis=0)-mu0),(X.mean(axis=0)-mu0))
    c=Decimal(np.linalg.det(eta0)**(v0/2))/(Decimal(np.linalg.det(etan))**((Decimal(v0)+n)/2))
    d=Decimal(kappa0/(kappa0+nf))**(d/2)
    return float(a*b*c*d)

## PIT, RIT calculatios between two nodes
def get_pi_ri(i,j,alpha):
    clust_k=i.combine(j,alpha)
    nk=len(clust_k.points)
    dk=clust_k.d
    pi=alpha*factorial(nk-1)/dk 
    all_points=list(clust_k.points)
    ph=prob_hypo(X[all_points][:],kappa0,v0,mu0,eta0) 
    pit=ph*pi+ (1-pi)*i.ph*j.ph
    rit=(pi*ph)/pit
    return pit,rit
    
    
## Function that calculates the maximum value in a dictionary, return key and max value
def get_dict_max(d):
    ind=0
    m=0
    for i in d:
        if d[i]>=m:
            m=d[i]
            ind=i
    return ind,m

## Get Node with the specific number from a list of nodes

def get_node(i,nodes):
    for node in nodes:
        if node.number==i:
            return node
    
## Incorporate pi,ri calculations go over all the nodes one inital time
def init2(X,alpha,kappa0,v0,mu0,eta0):
    x=[]
    for i in range(len(X)):
        node=Node(i,alpha,i)
        node.ph=prob_hypo(X[[i]],kappa0,v0,mu0,eta0)
        
        x.append(node)
    
    from scipy.special import factorial
    n=len(x)
    for i in range(n):
        for j in range(i+1,n):
            p,r=get_pi_ri(x[i],x[j],alpha)
            x[i].pit[j]=p
            x[i].rit[j]=r  
        
        
    return x

## Function that creates a new node and drops the nodes that give the highest pit value. Add the pit,ri values to the new node

def change_nodes(i,j,n,nodes,alpha):
 
    n1=get_node(i,nodes)
    n2=get_node(j,nodes)

    new_node=n1.combine(n2,alpha,i=n)
    new_node.ph=get_pi_ri(n1,n2,alpha=alpha)[0] 
    nodes.remove(n1)
    nodes.remove(n2)
    for node in nodes:
        
        new_node.pit[node.number],new_node.rit[node.number]=get_pi_ri(node,new_node,alpha=alpha)   
        ## nodes[k].pit[n],nodes[k].rit[n] =get_pi_ri(nodes[k],new_node,alpha=alpha) 
        if n1.number in node.rit:
            del node.rit[n1.number]
        if  n2.number in node.rit:
            del node.rit[n2.number]
    nodes.append(new_node)
    return nodes 

## Function for updating clusters, essentially this functions uses change_nodes function at each step by n times.
## We loop through all the nodes find the maximum probability combination and update the list.
## @nodes--the list of nodes , @n: number of times we merge the nodes(essentially depends on the number of clusters)

def update_clust(nodes,n,alpha):    
    ## Find the maximum rit
    m=0
    i=0
    j=0
    for k in range(len(nodes)):
        cind,cm=get_dict_max(nodes[k].rit)
        if cm>=m:
            m=cm
            i=nodes[k].number
            j=cind
    return change_nodes(i,j,n,nodes,alpha=alpha)

## The function that takex the data set and implements the clustering using the functions defined above
## @k--number of clusters
## Returns the labels, i.e clusters of all the data set and a list of nodes with k number of nodes.

def clust(X,alpha,mu0,eta0,kappa0=1,v0=2.5,k=3):
    n=len(X)
    nodes=init(X,alpha,kappa0,v0,mu0,eta0)
    y=np.zeros((n))
    for i in range(n-k):
        nodes=update_clust(nodes,n+i,alpha)
    ## Set the labels
    y=np.zeros(len(X),dtype="int32")
    for i in range(k):
        ind=list(nodes[i].points)
        y[ind]=i
    ## Finish the classification and produce the table
    for i in range(k-1) :
        nodes=update_clust2(nodes,n+i,alpha)
    return y,nodes