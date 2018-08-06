# FCN-pytorch-easiest

## trying to be the easiest FCN pytorch implementation and just in a get and use fashion
Here I use a handbag semantic segmentation for illustration on how to train FCN on your own dataset and just go to use.
To train on your own dataset you just need to see in ```BagData.py``` which implement a dataloader in pytorch. What you actually need to do is provide the images file and the correspoding mask images which I will discuss later. And for visualization in the training process I use ```visdom```.
