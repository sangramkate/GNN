import csv

filename = "cora-label.txt"

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

f = open(filename,'w')
with open('cora.content') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='	')
    line_count = 0
    for row in csv_reader:
        if line_count !=0 :
            val = 0;
            flag = 0;
            for label in label_name:
                if(row[len(row)-1] == label):
                    flag = 1
		    for i in label_value[val]:
                        f.write(str(i))
                    f.write("\n")
                    break
                val = val + 1
            if(flag == 0):
                #print("Not found: "+ row[len(row)-1] +":" +i ",".join(row) + "\n")
                print("Not found: "+ row[len(row)-1] +":" + "\n")
        line_count = line_count + 1
   

    f.close()         
