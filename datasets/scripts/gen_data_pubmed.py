#!usr/bin/python

import os
import re

features = {}

pubmed_content = open("../Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab","r")
pubmed_cites = open("../Pubmed-Diabetes/data/Pubmed-Diabetes.DIRECTED.cites.tab")

count = 0
for line in pubmed_content:
    if (re.match(".*summary=(.*)",line)):
        feat = re.match(".*summary=(.*)\n",line).group(1)
        feat_arr = feat.split(',')
        for key in feat_arr:
            if key not in features:
                features[key] = count 
                count = count + 1


print(count)
pubmed_content.close()

#print(features)
pubmed_content = open("../Pubmed-Diabetes/data/Pubmed-Diabetes.NODE.paper.tab","r")

data_hash = {}
node_count_map = {}
count = 0
for line in pubmed_content:
    if (re.match(".*label=.*",line)):
        content = line.split()
        node = str(content[0])
        data_hash[node] = {}
        node_count_map[node] = count
        #print(content)
        if(content[1] == "label=1"):
            data_hash[node]["label"] = [1,0,0]
        elif (content[1] == "label=2"):
            data_hash[node]["label"] = [0,1,0]
        elif (content[1] == "label=3"):
            data_hash[node]["label"] = [0,0,1]
        data_hash[node]["features"] = [0] * 500
        #print(content[-1])
        for feat in content[2:]:
            if re.search("(w-.*)=(.*)",feat):
                feat_name = re.search("(w-.*)=(.*)",feat).group(1) 
                val = re.search("(.*)=(.*)",feat).group(2)
                data_hash[node]["features"][features[feat_name]] = val
        #features_arr = content[-1].split('=')
        #if (features_arr[0] == "summary"):
        #    for feat in features_arr[1].split(','):
        #        data_hash[node]["features"][features[feat]] = 1
        data_hash[node]["edges"] = [int(count)] #added self loop
        count = count + 1

pubmed_content.close()
print(count)

edge = 0
for line in pubmed_cites:
    if (re.match(".*paper:(\d*).*paper:(\d*)",line)):
        cite = ["",""]
        cite[0] = re.match(".*paper:(\d*).*paper:(\d*)",line).group(1)
        cite[1] = re.match(".*paper:(\d*).*paper:(\d*)",line).group(2)
        data_hash[cite[0]]["edges"].append(int(node_count_map[cite[1]]))
        data_hash[cite[1]]["edges"].append(int(node_count_map[cite[0]]))
        edge = edge + 1

print(edge)


with open("../Pubmed-Diabetes/data_info.csv", "w") as info_file:
    info_file.write(str((count)) + ",")
    info_file.write(str((edge)) + ",")
    info_file.write("500,")
    info_file.write(str((3)) + "\n")


row_start = open("../Pubmed-Diabetes/row_start.csv","w")
edge_dst = open("../Pubmed-Diabetes/edge_dst.csv","w")
edge_data = open("../Pubmed-Diabetes/edge_data.csv", "w") #degree matrix inverse
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

with open("../Pubmed-Diabetes/edge_dst.csv", 'rb+') as filehandle:
    filehandle.seek(-1, os.SEEK_END)
    filehandle.truncate()

with open("../Pubmed-Diabetes/edge_data.csv", 'rb+') as filehandle:
    filehandle.seek(-1, os.SEEK_END)
    filehandle.truncate()

feat_val = open("../Pubmed-Diabetes/feature_val.csv","w")
label_val = open("../Pubmed-Diabetes/label_val.csv","w")
for node_count in sorted(node_count_map.items(), key=lambda x: x[1]):
    feat_count = len(data_hash[node_count[0]]["features"])
    #print(feat_count);
    for feat in range(0,feat_count):
        feat_val.write(str(data_hash[node_count[0]]["features"][feat]))
        if (feat != feat_count - 1):
            feat_val.write(",")
    feat_val.write("\n")

    if "label" not in data_hash[node_count[0]]:
        print(node_count[0])
    label_count = len(data_hash[node_count[0]]["label"])
    for lab in range(0,label_count):
        label_val.write(str(data_hash[node_count[0]]["label"][lab]))
        if(lab != label_count - 1):
            label_val.write(",")
    label_val.write("\n")

feat_val.close()
label_val.close()

#print(data_hash)
