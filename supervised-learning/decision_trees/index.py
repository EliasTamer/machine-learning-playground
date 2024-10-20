import numpy as np

'''
We will use one-hot encoding to encode the categorical features. They will be as follows:

Ear Shape: Pointy = 1, Floppy = 0
Face Shape: Round = 1, Not Round = 0
Whiskers: Present = 1, Absent = 0
Therefore, we have two sets:

X_train: for each example, contains 3 features:

      - Ear Shape (1 if pointy, 0 otherwise)
      - Face Shape (1 if round, 0 otherwise)
      - Whiskers (1 if present, 0 otherwise)
y_train: whether the animal is a cat

      - 1 if the animal is a cat
      - 0 otherwise
'''

X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])


def entropy(p):
    if p == 0 or p == 1:
        return 1
    else:
        return -p * np.log2(p) - (1 - p) * np.log2(1 - p)
    
    
def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have 
    that feature = 1 and the right node those that have the feature = 0 
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices


def weighted_entropy(X,y,left_indices,right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    
    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy

left_indices, right_indices = split_indices(X_train, 0)
weighted_entropy(X_train, y_train, left_indices, right_indices)


'''the weighted entropy in the 2 split nodes is 0.72.
to compute the information Gain we must subtract it from the entropy
in the node we chose to split (in this case, the root node).
'''

def information_gain(X, y, left_indices, right_indices):
    p_node = sum(y) / len(y)
    h_node = entropy(p_node)
    return h_node - weighted_entropy(X, y, left_indices, right_indices)

inf_gain = information_gain(X_train, y_train, left_indices, right_indices)


for i, feature_name in enumerate(['Ear Shape', 'Face shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    gain_i = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature {feature_name}: information gain if we split the root node using this feature:{gain_i:2f}")
    