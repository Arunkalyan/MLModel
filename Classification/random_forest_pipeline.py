#!/usr/bin/python

from math import sqrt

def randomForest_pipeline(total_features):
   
    pipe = Pipeline(steps=[ 
        ('clf', LogisticRegression())
        ])

    parameters = {'n_jobs': [-1],
                  'max_features': [sqrt(total_features)]
                  }

    return pipe, parameters
