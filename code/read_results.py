
import re
import os
import pandas as pd

os.chdir("directory")


##  15

files = os.listdir("directory")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_15/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()

##  25

files = os.listdir("masking_25")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_25/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_bert = pd.DataFrame(f1_scores).mean()
results_bert

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()


# 40

files = os.listdir("masking_40")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_40/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()



# 60

files = os.listdir("masking_60")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_60/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()


## 80 
files = os.listdir("masking_80")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_80/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_deberta = pd.DataFrame(f1_scores).mean()
results_deberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()



## 15 masked
files = os.listdir("masking_15_sel_mask")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_15_sel_mask/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()


## 25 masked
files = os.listdir("masking_25_sel_mask")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_25_sel_mask/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()



## 40 masked
files = os.listdir("masking_40_sel_mask")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_40_sel_mask/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()


## 60 masked
files = os.listdir("masking_60_sel_mask")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_60_sel_mask/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()


## 80 masked
files = os.listdir("masking_80_sel_mask")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_80_sel_mask/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()


## 15-40 masked
files = os.listdir("masking_15_40")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_15_40/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_roberta = pd.DataFrame(f1_scores).mean()
results_roberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()







## 40 masked
files = os.listdir("masking_40_sel_mask_deb")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_40_sel_mask_deb/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_deberta = pd.DataFrame(f1_scores).mean()
results_deberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()


## 40 masked
files = os.listdir("masking_40_deb")
files = [f for f in files if re.search("fakenews", f)]
files

f1_scores = []
for i in range(0, len(files)):
    file = open("masking_40_deb/" + files[i], "r")
    r = file.read().split(',')
    f1s = [f1 for f1 in r if re.search("f1\"", f1)]
    f1_scores.append([float(x.split(":")[1]) for x in f1s])
    
pd.DataFrame(f1_scores)
results_deberta = pd.DataFrame(f1_scores).mean()
results_deberta

pd.DataFrame(f1_scores).std()
pd.DataFrame(f1_scores).std().mean()