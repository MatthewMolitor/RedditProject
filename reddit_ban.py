#we only care about data from columns 1 and 6
import bisect

#preproccessing -- convert messages into features for classifier
with open("traindata.txt") as f:
    h = open("stoplist.txt", "r")
    vocabulary = []
    for i in f:
        for j in i.split():
            if j not in vocabulary:
                bisect.insort(vocabulary,j)
    for i in h:
        if i.rstrip() in vocabulary:
            vocabulary.remove(i.rstrip())
    h.close()

M = len(vocabulary)
D = {}
with open("traindata.txt") as f:
    x = 0
    for i in f:
        D[x] = [[0]*M,'label']
        for j in i.split():
            if j in vocabulary:
                D[x][0][vocabulary.index(j)] = 1
        x+=1

with open("trainlabels.txt") as f:
    x = 0
    for i in f:
        if i.rstrip() == "1":
            D[x][1] = 1
        else:
            D[x][1] = -1
        x+=1


testData = {}
with open("testdata.txt") as f:
    x = 0
    for i in f:
        testData[x] = [[0]*M,'label']
        for j in i.split():
            if j in vocabulary:
                testData[x][0][vocabulary.index(j)] = 1
        x+=1

with open("testlabels.txt", "r", encoding='utf-8-sig') as f:
    x = 0
    for i in f:
        if i.rstrip() == "1":
            testData[x][1] = 1
        else:
            testData[x][1] = -1
        x+=1


#dot product function modified from https://www.maxbartolo.com/ml-index-item/dot-scalar-product/
def dot(x, y):
    if sum(x_i*y_i for x_i, y_i in zip(x, y)) > 0:
        return 1
    else:
        return -1


nu = 1
T = 20
w_standard = [0]*M
w_average = [0]*M
mistakes_standard = []
mistakes_average = []

mistakes_test_stand = []
mistakes_test_avg = []

#standard perceptron
for i in range(T):
    mistakes_standard.append(0)
    mistakes_test_stand.append(0)
    for key in D.keys():
        if dot(D[key][0],w_standard) != D[key][1]:
            mistakes_standard[i]+=1
            for j in range(len(w_standard)):
                w_standard[j] += (nu*D[key][1]*D[key][0][j])
    for key in testData.keys():
        if dot(testData[key][0],w_standard) != testData[key][1]:
            mistakes_test_stand[i]+=1

c=1
#averaged perceptron
for i in range(T):
    mistakes_average.append(0)
    mistakes_test_avg.append(0)
    for key in D.keys():
        if dot(D[key][0],w_average) != D[key][1]:
            mistakes_average[i]+=1
            for j in range(len(w_average)):
                w_average[j] += (c*nu*D[key][1]*D[key][0][j])
        else:
            c = c + 1
    for key in testData.keys():
        if dot(testData[key][0],w_standard) != testData[key][1]:
            mistakes_test_avg[i]+=1







g = open("output.txt", "w")
k = 1
g.write("standard perceptron \n")
for i in mistakes_standard:
    g.write("iteration ")
    g.write(str(k))
    g.write(" no-of-mistakes:")
    g.write("\t")
    g.write(str(i))
    g.write("\n")
    k+=1

g.write("\ntraining accuracy vs test accuracy\n")
for i in range(20):
    g.write("iteration ")
    g.write(str(i+1))
    g.write(" Training Accuracy: ")
    g.write(str((len(D)-mistakes_standard[i])/len(D)))
    g.write("\tTesting Accuracy: ")
    g.write(str((len(D)-mistakes_test_stand[i])/len(D)))
    g.write("\n")

k=1
g.write("\naveraged perceptron \n")
for i in mistakes_average:
    g.write("iteration")
    g.write(str(k))
    g.write(" no-of-mistakes:")
    g.write("\t")
    g.write(str(i))
    g.write("\n")
    k+=1

g.write("\ntraining accuracy vs test accuracy\n")
for i in range(20):
    g.write("iteration-")
    g.write(str(i+1))
    g.write(" Training Accuracy: ")
    g.write(str((len(D)-mistakes_average[i])/len(D)))
    g.write("\tTesting Accuracy: ")
    g.write(str((len(D)-mistakes_test_avg[i])/len(D)))
    g.write("\n")

