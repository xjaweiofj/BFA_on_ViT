import math
import os
import numpy
import matplotlib.pyplot as plt

file_path = "./tiny_grad_seed42.log" 
file = open(file_path, "r")
content = file.readlines()
file.close()

idx = 0
read_w = 0
skip = 0

pct_dict = {}
flipped_w = []
flipped_w_all = [] # the most sensitive weight identified every layer, though most of them are not the flipped bit that was chosen finally

for line in content:
    #print (f"{idx}:{line}")
    idx += 1

    if "ite = " in line:
        ite = line.split()[2]
        print (f"ite {ite}:")
        skip = 0
    elif skip == 1:
        continue
    
    if "=========== print the weight gradient for layer " in line:
        layer_name = line.split()[-2]

    if "weight max=" in line:
        w_max = float(line.split("=")[1].split(",")[0])
        w_min = float(line.split("=")[-1])

    if "chosen_idx:" in line:
        read_w = 1
        continue
    if (read_w == 1):
        w_chosen = float(line.split("(")[1].split(")")[0])
        #w_topk_list = line.split("[")[1].split(".]")[0].split("., ")
        read_w = 0
        #w_topk_list = [float(element) for element in w_topk_list]
        #w_topk_max = max(w_topk_list)

        pct = abs(round(w_chosen/w_max*100, 3))
        pct_dict[layer_name] = pct

        flipped_w_all.append(pct)

    if "max_loss_module = " in line:
        chosen_layer = line.split()[2]
        for k, v in pct_dict.items():
            if (k == chosen_layer):
                flipped_w.append(pct)
                print (f"\tlayer {k}\tw_topk_max/w_max={v}% ===> the flipped bit is from this layer")
            else:
                print (f"\tlayer {k}\tw_topk_max/w_max={v}%")
        skip = 1

print (f"There are {len(flipped_w)} for the flipped weight pct")
print (f"There are {len(flipped_w_all)} for all flipped (candidate) weight pct")

# Plotting the histogram
plt.hist(flipped_w, bins=10, range=(0, 100))

# Setting labels and title
plt.xlabel('Magnitude of Data')
plt.ylabel('Amount of Data')
plt.title('Value Distribution of Flipped Weight (%)')

# Displaying the plot
plt.show()


# Plotting the histogram
plt.hist(flipped_w_all, bins=10, range=(0, 100))

# Setting labels and title
plt.xlabel('Magnitude of Data')
plt.ylabel('Amount of Data')
plt.title('Value Distribution of All Flipped Weight (including candidates that are not chosen) (%)')

# Displaying the plot
plt.show()
