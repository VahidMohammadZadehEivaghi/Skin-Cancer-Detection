# 🩺 Skin Cancer Detection (HAM10000 Dataset)

The HAM10000 dataset is a comprehensive collection of dermatoscopic images for skin lesion classification, widely used in the field of medical imaging and machine learning. It contains a diverse range of skin lesions, aimed at advancing research in dermatology, particularly for the diagnosis of skin cancers. The dataset consists of 10,000 high-resolution images of skin lesions, sourced from various individuals. This diversity helps in training robust machine learning models that can generalize well to unseen data. The primary challenge with the dataset is its significant **_imbalance_**. <br/>
Collected images are annotated and categorized into 7 classes including:
- Melanoma: 3
- Melanocytic Nevi: 1
- Basal Cell Carcinoma: 5
- Squamous Cell Carcinoma: 2
- Actinic Keratosis: 6
- Vascular Lesions: 4
- Benign Keratosis (Seborrheic Keratosis, etc.): 0

Source: Download the HAM1000 dataset from [here](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

# 💡 Class distribution 

In any machine learning endeavor, it is highly recommended to perform exploratory analysis before commencing the modeling phase. This process yields insights about the data, providing valuable information that can guide and improve the modeling efforts. For example, examining the distribution of categories within the HAM10000 dataset shows that it is imbalanced, necessitating careful strategies to address this issue.

<figure style="align: center;">
  <img src="https://github.com/user-attachments/assets/5d38fc8a-f527-486d-8aeb-0535092e0a3a" alt="Class distribution of HAM10000 dataset">

In such cases, giving equal weight to errors across all classes can cause problems, as the model might simply classify all instances as the majority class to achieve deceptively high accuracy. This approach ignores that different cancer types need different treatments. To address this issue, it is crucial to assign error weights to each class based on their populations. For example, if the nv class is the majority, the model should prioritize it less. I use the median frequency balancing for class weighting, though mode frequency balance and inverse frequency method are also options.

# 🥇 Finetuning ResNet50 

ResNet50, short for Residual Network with 50 layers, is a deep convolutional neural network designed to address the vanishing gradient problem in deep networks. Introduced by Kaiming He and colleagues in 2015, 
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385), ResNet50 leverages residual learning through skip connections, allowing the network to learn residual functions with reference to the input layers. This architecture enables the training of very deep networks, improving performance in tasks such as image classification, object detection, and more. <br/> 
Since the base model is trained on [ImageNet](https://www.image-net.org/) with 1000 classes, we need to replace its projection head with a customized layer for fine-tuning. The projection head, a simple linear classifier, takes inputs of dimension 2048 and outputs 7 class scores, followed by a softmax activation function to normalize these scores. There are two common approaches for training: either freeze the backbone from the base model and only train the new projection head, or adjust the parameters of the entire network. I opted for the latter because, without modifying the backbone parameters, the simple projection head cannot classify the data correctly. To fully utilize ResNet50's capabilities, the backbone parameters should not change significantly, so the learning rate for the backbone should be smaller than that for the projection head. <br/> 

The learning curve is reported as follows:

<figure style="align: center;">
  <img src="https://github.com/user-attachments/assets/1e781d12-f680-49dd-b649-173dade0d03c" alt="Learning curve">

And the accuracy plot:

<figure style="align: center;">
  <img src="https://github.com/user-attachments/assets/61a96220-8629-4d39-9cba-b65f210f66cf" alt="Accuracy">

It is important to note that for imbalanced datasets, relying solely on the accuracy metric can be misleading. Other metrics, such as precision and recall, should also be considered. The confusion matrix would be as the following:

<figure style="align: center;">
  <img src="https://github.com/user-attachments/assets/bc41ce50-09b3-4396-ac67-1a7bd27930bf" alt="Confusion matrix">
