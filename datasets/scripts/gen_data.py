#!usr/bin/python

import re
import os

cora_content = open("../cora/cora.content","r")
cora_cites = open("../cora/cora.cites","r")

label_name = ['Case_Based',
              'Genetic_Algorithms',
              'Neural_Networks',
              'Probabilistic_Methods',
              'Reinforcement_Learning',
              'Rule_Learning',
              'Theory']

label_value = [[1,0,0,0,0,0,0],
               [0,1,0,0,0,0,0],
               [0,0,1,0,0,0,0],
               [0,0,0,1,0,0,0],
               [0,0,0,0,1,0,0],
               [0,0,0,0,0,1,0],
               [0,0,0,0,0,0,1],
               ]


data_hash = {}
node_count_map = {}
count = 0
for line in cora_content:
    content = line.split("\t")
    node = str(content[0])
    data_hash[node] = {}
    node_count_map[node] = count
    features = content[1:1434]
    data_hash[node]["features"] = features
    label = re.match("(.*)\n",content[1434]).group(1)
    label_index = label_name.index(label) 
    data_hash[node]["label"] = label_value[label_index]
    data_hash[node]["edges"] = [int(count)] #added self loop
    count = count + 1

cora_content.close()
print(count)

edge = 0
for line in cora_cites:
    cite = line.split("\t")
    cite[1] = re.match("(.*)\n",cite[1]).group(1)
    data_hash[cite[0]]["edges"].append(int(node_count_map[cite[1]]))
    data_hash[cite[1]]["edges"].append(int(node_count_map[cite[0]]))
    edge = edge + 1

print(edge)



#print(data_hash)
#print(node_count_map)
nnz_data = open("../cora/nnz_data.csv","w")
nnz_count = edge*2+count
for i in range(0,nnz_count):
    nnz_data.write(str(1))
    if(i != nnz_count - 1):
        nnz_data.write(",")
nnz_data.close()

row_start = open("../cora/row_start.csv","w")
edge_dst = open("../cora/edge_dst.csv","w")
edge_data = open("../cora/edge_data.csv", "w") #degree matrix inverse
count = 0
row_start.write(str(0))
for node_count in sorted(node_count_map.items(), key=lambda x: x[1]):
    count = count + len(data_hash[node_count[0]]["edges"]) 
    degree = 1.0/len(data_hash[node_count[0]]["edges"])
    row_start.write("," + str(count))
    #print(node_count[0] is None)
    data_hash[node_count[0]]["edges"].sort()
    #print(data_hash[node_count[0]]["edges"])
    for num in data_hash[node_count[0]]["edges"]:
        edge_dst.write(str(num) + ",")
        edge_data.write(str(degree) + ",")

edge_dst.close()
row_start.close()
edge_data.close()

with open("../cora/edge_dst.csv", 'rb+') as filehandle:
    filehandle.seek(-1, os.SEEK_END)
    filehandle.truncate()

feat_val = open("../cora/feature_val.csv","w")
label_val = open("../cora/label_val.csv","w")
for node_count in sorted(node_count_map.items(), key=lambda x: x[1]):
    feat_count = len(data_hash[node_count[0]]["features"])
    #print(feat_count);
    for feat in range(0,feat_count):
        feat_val.write(str(data_hash[node_count[0]]["features"][feat]))
        if (feat != feat_count - 1):
            feat_val.write(",")
    feat_val.write("\n")

    label_count = len(data_hash[node_count[0]]["label"])
    for lab in range(0,label_count):
        label_val.write(str(data_hash[node_count[0]]["label"][lab]))
        if(lab != label_count - 1):
            label_val.write(",")
    label_val.write("\n")

feat_val.close()
label_val.close()

#print(data_hash)
