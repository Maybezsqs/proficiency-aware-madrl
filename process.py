# -*- coding: utf-8 -*-

import pickle#, pprint
import argparse

parser = argparse.ArgumentParser("Read data points for Mix-DRL")
parser.add_argument("--pkl-dir", type=str, default="", help="directory in which .pkl is saved")
parser.add_argument("--split", type=int, default=0, help="process the data for each agent")
arglist = parser.parse_args()

pkl_file = open(arglist.pkl_dir, 'rb')
data = pickle.load(pkl_file) # data list

# Screen print
#pprint.pprint(data)
#print("length of data is " + str(len(data)))

if arglist.split != 0:
    # Agent rewards
    for i in range(arglist.split):
        f = open(arglist.pkl_dir.rstrip(".pkl")+"-"+str(i)+".txt","w") # w+
        f.write(str(data[i::4]))
        f.close()
else:
    # Total rewards
    f = open(arglist.pkl_dir.rstrip("pkl")+"txt","w")
    f.write(str(data))
    f.close()