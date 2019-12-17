import pprint, pickle
import argparse

parser = argparse.ArgumentParser("Read data points for MADRL")
parser.add_argument("--pkl-dir", type=str, default="", help="directory in which .pkl is saved")
arglist = parser.parse_args()

pkl_file = open(arglist.pkl_dir, 'rb')
data = pickle.load(pkl_file)

# Screen print
#pprint.pprint(data)
#print("length of data is " + str(len(data)))

fw = open(arglist.pkl_dir.rstrip("pkl")+"txt","w") # w+
fw.write(str(data))
fw.close()
