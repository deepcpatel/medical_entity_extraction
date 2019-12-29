# Training DNC

import sys
import time
import torch
import random
import numpy as np

# torch.autograd.set_detect_anomaly(True) # Setting Anomaly Detection True for finding bad operations

####### Following function is adapted from the implemention by loudinthecloud on Github ##########
def random_seed():
    seed = int(time.time()*10000000)
    random.seed(seed)
    np.random.seed(int(seed/10000000))      # NumPy seed Range is 2**32 - 1 max
    torch.manual_seed(seed)
##########################################################################################################

def main():
    if len(sys.argv) > 3:
        if sys.argv[1] == "GPU":
            if torch.cuda.is_available():  # Checking if GPU Request is given or not and availability of CUDA
                from tasks.ner_task_bio_GPU import task_NER
            else:
                print("GPU not found!")
                exit()
        elif sys.argv[1] == "CPU":
            from tasks.ner_task_bio import task_NER
        else:
            print("Please specify the run device (GPU/CPU)")
            exit()
        c_task = task_NER()                    # Initialization of the NER Task (This is for BIO Tagging. Edit at line 33 and 35 for BIEOS tagging)
        print("\nStarting Medical NER Task for DNC\n")
    else:
        print("Incorrect Number of arguments")
        exit()

    epoch = sys.argv[2] # Last Epoch number till the model was trained (eg: 0)
    batch = sys.argv[3] # Last Batch Number till the model was trained (eg: 1000)
    batch_size = 1

    # Random Seed
    random_seed()

    c_task.init_dnc()
    c_task.init_loss()
    c_task.batch_size = batch_size
    c_task.load_model(2, epoch, batch)
    results = c_task.test_model()

    print(results)

if __name__ == '__main__':
    main()