import os
import torch
# TODO: Generate folder with dict of minima and maxima and full input/labels
# transform them s.t. they are between 0 and 1

DATA_PER_SET = 501

def generate_tensor_dataset(shape):
    path = "new_data" + os.sep + "data_"+str(shape[0])+"x"+str(shape[1])
    for k in range(DATA_PER_SET)

def main():
    pass

if __name__ == "__main__":
    main()