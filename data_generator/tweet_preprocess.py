import pandas as pd
import numpy as np

df = pd.read_csv('data/training.1600000.processed.noemoticon.csv',
                 encoding = 'latin',header=None)
df = df[[5,0]]
lab_to_sentiment = {0:0, 4:1}
def label_decoder(label):
  return lab_to_sentiment[label]
df[0] = df[0].apply(lambda x: label_decoder(x))

df.to_csv( 'data/twitter_sentiment140.csv', index=False, header=False )