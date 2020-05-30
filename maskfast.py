from fastai import *
from fastai.vision import *
import pathlib


path = "/home/phaneendra/Downloads/dataset/Mask_Datasets/"

tfms = get_transforms(do_flip=False)

data = ImageDataBunch.from_folder(path,ds_tfms=tfms,size=224,bs=16)



def trainlearner():
    
    data.show_batch(rows=3,figsize=(5,5))

    size = 224

    learn = create_cnn(data,models.resnet34,metrics=error_rate)

    learn.model_dir='/home/phaneendra/Documents/python/fastai'

    learn.fit_one_cycle(4)

    learn.save('mask-final')

    learn.unfreeze()

    learn.lr_find()

    learn.recorder.plot()

    interp = ClassificationInterpretation.from_learner(learn)

    interp.plot_confusion_matrix()


def testlearner():
    
    data = ImageDataBunch.from_folder(path,ds_tfms=tfms,size=224)
    
    learn = cnn_learner(data,models.resnet34).load('/home/phaneendra/Documents/python/fastai/mask-final')

    img = open_image('46.jpg')

    prediction,pred_idx,output = learn.predict(img)

    print(prediction,pred_idx,output)

if __name__ == "__main__":
    trainlearner()
    testlearner()

    
    

