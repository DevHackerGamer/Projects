import pandas as pd
import numpy as np
from collections import Counter

class Node:
    def __init__(self,feature=None,threshold=None,left=None,right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:

    def __init__(self,minSampleSplit=2,maxDepth=100,numfeatures=None):
        self.minSampleSplit=minSampleSplit
        self.maxDepth=maxDepth
        self.numfeatures=numfeatures
        self.root=None

    def fit(self,x,y):
        x = x.values
        y = y.values.ravel()
        self.numfeatures = x.shape[1] if not self.numfeatures else min(x.shape[1],self.numfeatures) 
        self.root = self._grow_tree(x,y,depth=0) #look here

    def _grow_tree(self,x,y,depth):
        numSamples,numfeature = x.shape
        numLabels = len(np.unique(y))

        # check for stopping critetia
        if(depth>=self.maxDepth or numLabels==1 or numSamples <self.minSampleSplit):
            leafValue = self._most_common_label(y)
            return Node(value=leafValue)
        
        feat_idxs = np.random.choice(numfeature,self.numfeatures,replace=False)        
        # find best split
        bestFeature,bestThreshold = self._best_split(x,y,feat_idxs)

        if bestFeature is None:
            leafValue = self._most_common_label(y)
            return Node(value=leafValue)

        # create child nodes
        left_idxs,right_idxs = self._split(x[:,bestFeature],bestThreshold)
        left = self._grow_tree(x[left_idxs,:],y[left_idxs],depth+1)
        right = self._grow_tree(x[right_idxs,:],y[right_idxs],depth+1)
        return Node(bestFeature,bestThreshold,left,right)

    def _best_split(self,x,y,feat_idxs):
        bestGain = -1
        split_idx , splitThreshold = None,None

        for feat_idx in feat_idxs:
            xCol = x[:,feat_idx]
            thresholds = np.unique(xCol)

            for thresh in thresholds:
                gain = self._information_gain(y,xCol,thresh)

                if gain>bestGain:
                    bestGain = gain
                    split_idx = feat_idx
                    splitThreshold =thresh
        return split_idx , splitThreshold

    def _information_gain(self,y,xCol,threshold):
        # parent enropypy
        parentEntropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(xCol,threshold)
        if(len(left_idxs)==0 or len(right_idxs)==0):
            return 0
        
        # claculate weigt avg entropy of childen
        n = len(y)
        n_l ,n_r = len(left_idxs) , len(right_idxs)
        e_l,e_r = self._entropy(y[left_idxs]),self._entropy(y[right_idxs])
        childEntropy = (n_l/n)*e_l+(n_r/n)*e_r

        # calculate IG
        informationGain = parentEntropy-childEntropy
        return informationGain

    def _split(self,xCol,splitThreshold):
        left_idxs = np.argwhere(xCol<=splitThreshold).flatten()
        right_idxs = np.argwhere(xCol>splitThreshold).flatten()
        return left_idxs,right_idxs

    def _entropy(self,y,esp=1e-10):
        hist = Counter(y)
        ps = [count/len(y) for count in hist.values()]
        return -np.sum([p*np.log(p+esp)for p in ps if p>0])

    def _most_common_label(self,y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value
    
    def predict(self,X):
        if isinstance(X,pd.DataFrame):
            X = X.values
        return np.array([self._traverse_tree(x,self.root) for x in X])
    
    def _traverse_tree(self,x,node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature]<=node.threshold:
            return self._traverse_tree(x,node.left)
        return self._traverse_tree(x,node.right)