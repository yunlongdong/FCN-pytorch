# FCN-pytorch-easiest

### Trying to be the easiest FCN pytorch implementation and just in a get and use fashion
Here I use a handbag semantic segmentation for illustration on how to train FCN on your own dataset and just go to use.
To train on your own dataset you just need to see in ```BagData.py``` which implements a dataloader in pytorch. What you actually need to do is providing the images file and the correspoding mask images. And for visualization in the training process I use ```visdom```.

### requirement

I have tested the code in ```pytorch 0.3.0.post4``` in anaconda ```python 3.6 ``` in ```ubuntu 14.04``` with ```GTX1080``` in ```cuda8.0``` 

### train

here three images pair is provided in folder ```last/``` and ```last_msk/``` . Here I want to do a handbag semantic segmentation which is stated as belows.

![task](https://github.com/yunlongdong/FCN-pytorch-easiest/blob/master/images/task.png)

Firstly because ```visdom``` is used to visualize the training process, you need open another terminal and run 

```sh
python -m visdom.server
```

Then you run in another terminal

```sh
python FCN.py
```

You can open your browser and goto ```localhost:8097``` to see the visulization as following the first row is the prediction.

![vis](https://github.com/yunlongdong/FCN-pytorch-easiest/blob/master/images/vis.png)

### deploy
and for deploy and inference I also provide a script ```inference.py```. You should be careful about the model path. Bacause I did not provide the trained weights file. :-P

BTW, ```FCN.py``` is copy from other repo.
