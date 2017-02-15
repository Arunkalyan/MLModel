#!/usr/bin/python

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=num_features_for_split, n_jobs=-1)
rf.fit(df_train.ix[:, df_train.columns != 'label'],df_train['label'])
#rf.predict(df_test.head(10).ix[:, df_test.head(10).columns != 'label'])
rf.score(df_test.ix[:, df_test.columns != 'label'],df_test['label'])
