import splitfolders

input_folder = 'flowers/'

# Split with a ratio
# To only split into training and validation set, set a tuple to 'ratio', i,e,
# Train, validation, test

splitfolders.ratio(input_folder, output='Flowers Train_Valid_Test',
                   seed=42, ratio=(0.7, 0.2, 0.1),         # train=0.7,valid=0.2,test=0.1
                   group_prefix=None)