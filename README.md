# ERA-S8-Cifar-10-Models

## Objective 
To write 3 models each with Batch Normalization, Layer Normalization and Group Normalization and train it on CIFAR 10 data.
Each model should achieve at least 70% validation accuracy. The base structure of the models are given.

## Steps Followed
- Data Downloaded
- Mean and std found for each of the channels are found
- normalization is donr accordingly
- model is built according to the structure
- trained and tested for 20 epochs
- ran inference to get misclassified images
- plotted the images and graphs

## Model stats
- BN Model
  - Params: 36664
  - Val Accuracy: 76.19%
- LN Model
  - Params: 36664
  - Val Accuracy: 76%
- GN Model
  - Params: 36664
  - Val Accuracy: 76.44%

## Learnings
- All the models have the same parameters.
- Increasing number of groups increased val accuracy.

## Misclassified images
`Batch Normalization`

![misclassified_images_bn](https://github.com/kurchi1205/ERA-S8-Cifar-10-Models/assets/40196782/7f9074bf-3865-4c79-8b4b-6541ada46906)



`Layer Normalization`

![misclassified_images_ln](https://github.com/kurchi1205/ERA-S8-Cifar-10-Models/assets/40196782/a55b03e5-b653-4aa6-962d-a108fb9e58cd)



`Group Normalization`

![misclassified_images_gn](https://github.com/kurchi1205/ERA-S8-Cifar-10-Models/assets/40196782/5c1b15d2-9b72-46ff-988a-225384dccf90)


## Graphs
`Batch Normalization`

<img width="896" alt="Screen Shot 2023-06-24 at 8 11 15 AM" src="https://github.com/kurchi1205/ERA-S8-Cifar-10-Models/assets/40196782/51430ccb-4fa7-4408-b526-25bd3b0facc8">

`Layer Normalization`

<img width="892" alt="Screen Shot 2023-06-24 at 8 12 26 AM" src="https://github.com/kurchi1205/ERA-S8-Cifar-10-Models/assets/40196782/819350cb-003f-4a79-9e92-0b0f63bf2354">

`Group Normalization`

<img width="881" alt="Screen Shot 2023-06-24 at 8 13 07 AM" src="https://github.com/kurchi1205/ERA-S8-Cifar-10-Models/assets/40196782/5918fccb-89a3-43ec-a72c-f3c3704e79b7">
