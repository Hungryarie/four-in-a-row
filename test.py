import numpy as np
#https://stackoverflow.com/questions/13546521/rotating-rectangular-array-by-45-degrees


import pandas as pd
"""
bd = np.matrix([[44., -1., 40., 42., 40., 39., 37., 36., -1.],
                [42., -1., 43., 42., 39., 39., 41., 40., 36.],
                [37., 37., 37., 35., 38., 37., 37., 33., 34.],
                [35., 38., -1., 35., 37., 36., 36., 35., -1.],
                [36., 35., 36., 35., 34., 33., 32., 29., 28.],
                [38., 37., 35., -1., 30., -1., 29., 30., 32.]])
"""
bd=np.matrix([[0.,0.,0.,0.,0.,4.,1.],
              [0.,1.,0.,0.,2.,0.,4.],
              [0.,0.,1.,2.,0.,0.,3.],
              [0.,0.,2.,1.,0.,3.,0.],
              [0.,2.,0.,0.,1.,0.,0.],
              [5.,1.,0.,3.,0.,0.,0.]])
bd=np.matrix([[0,0,0,0,0,4,1],
              [0,1,0,0,2,0,4],
              [0,0,1,2,0,0,3],
              [0,0,2,1,0,3,0],
              [0,2,0,0,1,0,0],
              [5,1,0,3,0,0,0]])

def NProtate45(array):
    rows, cols = array.shape
    #print (f"rows:{rows}, cols:{cols}")
    rot = np.zeros([rows,cols+rows-1],dtype=int)
    #print (rot)
    #print (array)
    for i in range(rows):
        for j in range(cols):
            rot[i,i + j] = array[i,j]
    return rot

def NProtate275(array):
    rows, cols = array.shape
    #print (f"rows:{rows}, cols:{cols}")
    rot = np.zeros([rows,cols+rows-1],dtype=int)
    #print (rot)
    #print (array)
    for i in range(rows):
        for j in range(cols):
            rot[i,i - j] = array[i,j]
    return rot

print (NProtate45(bd))
print("")
print (NProtate275(bd))

def rotate45(array):
    rot = []
    for i in range(len(array)):
        rot.append([0] * (len(array)+len(array[0])-1))
        for j in range(len(array[i])):
            rot[i][int(i + j)] = array[i][j]
    return rot

#df_bd = pd.DataFrame(data=np.matrix(rotate45(bd.transpose().tolist())))
#df_bd = df_bd.transpose()
#print (df_bd)
#df_bd = pd.DataFrame(data=np.matrix(rotate45(bd.tolist())))
#df_bd = df_bd.transpose()
#print (df_bd)

print (np.matrix(rotate45(bd.tolist())))
print("")
#print (np.matrix(rotate45(bd.transpose().tolist())))

def rotate275(array):
    rot = []
    for i in range(len(array)):
        rot.append([0] * (len(array)+len(array[0])-1))
        for j in range(len(array[i])):
            rot[i][int(i - j)] = array[i][j]
    return rot

#df_bd = pd.DataFrame(data=np.matrix(rotate275(bd.transpose().tolist())))
#df_bd = df_bd.transpose()
#print (df_bd)
print (np.matrix(rotate275(bd.tolist())))