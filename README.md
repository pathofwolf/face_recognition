Simple CNN model to identify images from CelebA datates.

I made ordinary user interface using PySide6. You can upload a photo into a special window and then a person's photo will be detected in the photo. The face will be highlighted with a red square and the probability that the belongs to a certain class(man or woman) will be given.

First model provide accuracy = 0.9191 and loss = 0.1975 on validation data.

Second model provide accuracy = 0.9409 and loss = 0.1609 on validation data.

I used data augmentation. Also, I tried to learn model based on ResNet50 in 2 epoches, but i abandoned this idea, because we need more epoches which will require long training. Also, I would like to use GPU, but I have AMD radeon graphics and used tensorflow.
