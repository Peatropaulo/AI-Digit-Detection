import inspect
import sys
import math
import numpy as np

feature_set = {}
not_feature_set = {}
prob_y = {}

feature_set_p = {}
not_feature_set_p= {}

feature_set_h = {}
not_feature_set_h= {}
feature_set_l = {}
not_feature_set_l= {}
set_feat = 4;



'''
Raise a "not defined" exception as a reminder 
'''
def _raise_not_defined():
    print "Method not implemented: %s" % inspect.stack()[1][3]
    sys.exit(1)


'''
Extract 'basic' features, i.e., whether a pixel is background or
forground (part of the digit) 
'''
def extract_basic_features(digit_data, width, height):
    features=[]
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here
    for x in range(0, width):
        temp = []
        for y in range(0, height):
            if not digit_data[x][y]:
                temp.append(False)
            else:
                temp.append(True)
        features.append(temp)
    return np.array(features)

'''
Extract advanced features that you will come up with 
'''
def extract_advanced_features(digit_data, width, height):
    features2=[]
    features1=[]
    advanced_3=[]
    
    for x in range(0,width):
        temp1 = []
        temp2 = []
        for y in range(0,height):
            if digit_data[x][y] ==2:
                temp1.append(True)
                temp2.append(False)
            elif digit_data[x][y]==1:
                temp1.append(False)
                temp2.append(True)
            else:
                temp1.append(False)
                temp2.append(False)
        features2.append(temp2)
        features1.append(temp1)
    advanced_3.append(np.array(features1));
    advanced_3.append(np.array(features2));
    advanced_3.append(extract_basic_features(digit_data,width,height));
    if set_feat == 1:
        return extract_basic_features(digit_data,width,height)
    if set_feat == 2:
        return np.array(features1);
    if set_feat == 3:
        return np.array(features2);
    return advanced_3



'''
    else:    
    
Extract the final features that you would like to use
'''
def has_loop(digit_data,width ,height, startx,starty):
    if(startx+1<width and starty-1>=0 and startx-1>=0 and starty+1<height):
        if(digit_data[startx][starty] == digit_data[startx+1][starty] and digit_data[startx][starty] == digit_data[startx-1][starty]
            and digit_data[startx][starty] == digit_data[startx][starty+1]and digit_data[startx][starty] == digit_data[startx][starty-1]):
            return True
    return False


def extract_final_features(digit_data, width, height):
    features2=[]
    features1=[]
    features3=[]
    advanced_3=[]

    for x in range(0,width):
        temp1 = []
        temp2 = []
        temp3 = []
        for y in range(0,height):
            if digit_data[x][y] ==2:
                temp1.append(True)
                temp2.append(False)
            elif digit_data[x][y]==1:
                temp1.append(False)
                temp2.append(True)
            else:
                temp1.append(False)
                temp2.append(False)
            if has_loop(digit_data,width,height,x,y):
                temp3.append(True)
            else:
                temp3.append(False)
        features2.append(temp2)
        features1.append(temp1)
        features3.append(temp3)
    advanced_3.append(np.array(features1));
    advanced_3.append(np.array(features2));
    advanced_3.append(np.array(features3));
    
    return advanced_3
    

'''
Compupte the parameters including the prior and and all the P(x_i|y). Note
that the features to be used must be computed using the passed in method
feature_extractor, which takes in a single digit data along with the width
and height of the image. For example, the method extract_basic_features
defined above is a function than be passed in as a feature_extractor
implementation.

The percentage parameter controls what percentage of the example data
should be used for training. 
'''

def compute_statistics(data, label, width, height, feature_extractor, percentage=100.0):
    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here
    global prob_y
    global set_feat
    global feature_set
    global not_feature_set
    if set_feat == 0:
        num = int(math.floor(len(data)*(percentage/100.0)))
        new_label = {}
        for l in range(0, num):
            name = str(label[l])
            if name not in feature_set_h:
                feature_set_h[name] = np.zeros((width, height))
                if name not in feature_set_p:
                    feature_set_p[name] = np.zeros((width, height))
                new_label[name] = 0
                prob_y[name] = 0

            feat_set = feature_extractor(data[l], width, height)
            feature_set_h[name] =  feat_set[1] + feature_set_h[name]
            feature_set_p[name] =  feat_set[0] + feature_set_p[name]
            new_label[name] = 1 + new_label[name]
        for i in new_label:
            not_feature_set_h[i] = np.zeros((width, height))
            not_feature_set_p[i] = np.zeros((width, height))
            for x in range(0, width):
                for y in range(0, height):
                    not_feature_set_h[i][x][y] = math.log10((new_label[i]-feature_set_h[i][x][y]+1)) - math.log10((new_label[i]+1))
                    not_feature_set_p[i][x][y] = math.log10((new_label[i]-feature_set_p[i][x][y]+1)) - math.log10((new_label[i]+1))
                    feature_set_h[i][x][y] = math.log10((feature_set_h[i][x][y]+1)) - math.log10((new_label[i]+1))
                    feature_set_p[i][x][y] = math.log10((feature_set_p[i][x][y]+1)) - math.log10((new_label[i]+1))
            prob_y[i] = math.log10(float(new_label[i])/float(num))
    elif set_feat == 1 or set_feat == 2 or set_feat == 3:
        num = int(math.floor(len(data)*(percentage/100.0)))
        new_label = {}
        for l in range(0, num):
            name = str(label[l])
            if name not in feature_set:
                feature_set[name] = np.zeros((width, height))
                new_label[name] = 0
                prob_y[name] = 0
            feature_set[name] = feature_extractor(data[l], width, height) + feature_set[name]
            new_label[name] = 1 + new_label[name]
        for i in new_label:
            not_feature_set[i] = np.zeros((width, height))
            for x in range(0, width):
                for y in range(0, height):
                    not_feature_set[i][x][y] = math.log10((new_label[i]-feature_set[i][x][y]+1)) - math.log10((new_label[i]+1))
                    feature_set[i][x][y] = math.log10((feature_set[i][x][y]+1)) - math.log10((new_label[i]+1))
            prob_y[i] = math.log10(float(new_label[i])/float(num))
    else:
        num = int(math.floor(len(data)*(percentage/100.0)))
        new_label = {}
        for l in range(0, num):
            name = str(label[l])
            if name not in feature_set_h:
                feature_set_h[name] = np.zeros((width, height))
                if name not in feature_set_p:
                    feature_set_p[name] = np.zeros((width, height))
                if name not in feature_set_l:
                    feature_set_l[name] = np.zeros((width, height))
                new_label[name] = 0
                prob_y[name] = 0

            feat_set = feature_extractor(data[l], width, height)
            feature_set_h[name] =  feat_set[1] + feature_set_h[name]
            feature_set_p[name] =  feat_set[0] + feature_set_p[name]
            feature_set_l[name] =  feat_set[2] + feature_set_p[name]
            new_label[name] = 1 + new_label[name]
        for i in new_label:
            not_feature_set_h[i] = np.zeros((width, height))
            not_feature_set_p[i] = np.zeros((width, height))
            not_feature_set_l[i] = np.zeros((width, height))
            for x in range(0, width):
                for y in range(0, height):
                    not_feature_set_h[i][x][y] = math.log10((new_label[i]-feature_set_h[i][x][y]+1)) - math.log10((new_label[i]+1))
                    not_feature_set_p[i][x][y] = math.log10((new_label[i]-feature_set_p[i][x][y]+1)) - math.log10((new_label[i]+1))
                    not_feature_set_l[i][x][y] = math.log10((new_label[i]-feature_set_l[i][x][y]+1)) - math.log10((new_label[i]+1))
                    feature_set_l[i][x][y] = math.log10((feature_set_l[i][x][y]+1)) - math.log10((new_label[i]+1))
                    feature_set_h[i][x][y] = math.log10((feature_set_h[i][x][y]+1)) - math.log10((new_label[i]+1))
                    feature_set_p[i][x][y] = math.log10((feature_set_p[i][x][y]+1)) - math.log10((new_label[i]+1))
            prob_y[i] = math.log10(float(new_label[i])/float(num))


'''
For the given features for a single digit image, compute the class 
'''
def compute_class(features):
    predicted = -1

    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here
    global feature_set
    global prob_y
    global not_feature_set

    if set_feat ==0:
        arg = 0
        temp1 = 0
        prob = 0
        width = len(features)
        length = len(features[0])
        temp = np.zeros((width, length))
        for i in prob_y:
            temp = np.multiply(feature_set_h[i], features)
            temp = temp + np.multiply(not_feature_set_h[i], abs(features[1]-1))
            temp2 = np.multiply(feature_set_p[i], features)
            temp2 = temp2 + np.multiply(not_feature_set_p[i], abs(features[0]-1))
            prob = sum(sum(temp) + sum(temp2)) + prob_y[i]
            temp1 = 10**prob
            if temp1 > arg:
                arg = temp1
                predicted = i
    elif set_feat != 4:    
        arg = 0
        temp1 = 0
        prob = 0
        width = len(features)
        length = len(features[0])
        temp = np.zeros((width, length))
        for i in prob_y:
            temp = np.multiply(feature_set[i], features)
            temp = temp + np.multiply(not_feature_set[i], abs(features-1))
            prob = sum(sum(temp)) + prob_y[i]
            temp1 = 10**prob
            if temp1 > arg:
                arg = temp1
                predicted = i
    else:
        arg = 0
        temp1 = 0
        prob = 0
        width = len(features)
        length = len(features[0])
        temp = np.zeros((width, length))
        for i in prob_y:
            temp = np.multiply(feature_set_h[i], features)
            temp = temp + np.multiply(not_feature_set_h[i], abs(features[1]-1))
            temp2 = np.multiply(feature_set_p[i], features)
            temp2 = temp + np.multiply(not_feature_set_p[i], abs(features[0]-1))
            temp3 = np.multiply(feature_set_l[i], features)
            temp3 = temp2 + np.multiply(not_feature_set_l[i], abs(features[2]-1))
            prob = sum(sum(temp)) + prob_y[i]
            temp1 = 10**prob
            if temp1 > arg:
                arg = temp1
                predicted = i
        
    return int(predicted)

'''
Compute joint probaility for all the classes and make predictions for a list
of data
'''
def classify(data, width, height, feature_extractor):

    predicted=[]

    # Your code starts here 
    # You should remove _raise_not_defined() after you complete your code
    # Your code ends here
    size = len(data)
    for num in range(0, size):
        predicted.append(compute_class(feature_extractor(data[num], width, height)))

    return predicted







        
    
