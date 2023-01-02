# A study of backbone models
## ILSVRC Evaluation
### goal
- get severial ImageNet pre-trained weights
- evaluate top1 and top5 accuracy on ILSVRC 2012 validation set
### prepare dataset
- Download full dataset from [ILSVRC kaggle](https://www.kaggle.com/competitions/imagenet-object-localization-challenge/data)
- under Data/CLS-LOC contain following folders
  - train: 
  - val: 50000 images
  - test: 100000 images
- after unzip, there is a csv file named LOC_val_solution 
### result
- model: [mobilenet_v2](https://keras.io/api/applications/mobilenet/) trained from keras application
```
python mnv2_evaluation.py
```
