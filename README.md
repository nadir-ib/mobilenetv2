#Mobilenet-v2
Using Keras MobileNet-v2 model for your custom images dataset

Your custom dataset needs to have the following structure: for every class, create a folder containing .jpg sample images:

```
dataset_directory\
    class1\
        img1.jpg
        img2.jpg
    class2\
        img1.jpg
        img2.jpg
```


## How to use it?

1. Configure the parameters in config.json
2. Train the model using `python train.py.`
3. Evaluate the model on the test dataset using: `python test.py.``
