Link to dataset: https://www.kaggle.com/datasets/iamsouravbanerjee/animal-image-dataset-90-different-animals


![Animal Detection](https://i.imgur.com/oNYGSIA.png)

# Image Classification

[![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Benji1155/Image-Classification)](https://img.shields.io/github/v/release/Benji1155/Image-Classification)
[![GitHub last commit](https://img.shields.io/github/last-commit/Benji1155/Image-Classification)](https://img.shields.io/github/last-commit/Benji1155/Image-Classification)
[![GitHub issues](https://img.shields.io/github/issues-raw/Benji1155/Image-Classification)](https://img.shields.io/github/issues-raw/Benji1155/Image-Classification)
[![GitHub pull requests](https://img.shields.io/github/issues-pr/Benji1155/Image-Classification)](https://img.shields.io/github/issues-pr/Benji1155/Image-Classification)
[![GitHub](https://img.shields.io/github/license/Benji1155/Image-Classification)](https://img.shields.io/github/license/Benji1155/Image-Classification)

**Introduction**

In this assessment, the task was to develop multiple CNN and Deep Learning networks to classify a set of images. This report will detail the process, decisions made during the development of the networks, and comparisons between their performance.

**Images**

![K-Longo](https://i.imgur.com/pA0MkXC.png)

![K-Longo](https://i.imgur.com/geT7n85.png)

**Dataset/Preparation**

The dataset consisted of 90 animal classes with around 5,000 images. These images were organized into class folders for easy navigation. Since the images varied in size, they needed to be resized to a uniform size for optimal training. Initially, images were resized to 256x256, but this resulted in slow training times. A smaller size of 128x128 was used instead, providing faster training with good results.

To ensure data integrity, I implemented a check for duplicate images, as duplicates can cause imbalances and affect training accuracy. No duplicates were found. For data splitting, I used train_test_split from sklearn.model_selection. Initially, a split of 80-10-10 (train-test-validation) was used, but this was adjusted to 60-20-20 for better model performance.

**One Convolutional Layer**

When developing the first CNN network, I used a 256x256 image size, but this caused the model training to be very slow. A smaller size of 128x128 was chosen, which improved the training time without sacrificing performance. Batch size was a key parameter, and after experimenting with values between 32 and 256, a batch size of 128 was found to be optimal for stability and accuracy.

The first model achieved a training accuracy of 0.99, but the test and validation accuracy were much lower, at 0.28 and 0.26, respectively. The high training accuracy indicates overfitting, as the model was unable to generalize well to unseen data. The performance also showed a zigzag pattern, suggesting that the model was not improving after a certain point.

**Two Convolutional Layers**

Increasing the complexity by adding a second convolutional layer did not result in significant improvement. The validation accuracy remained around 0.28, the test accuracy at 0.27, and the training accuracy remained high at 0.99. Trying different activation functions, such as changing from ReLU to Sigmoid, did not help, as Sigmoid is better suited for binary classification problems. Autotune was introduced, which resolved issues where the model would often encounter near-zero accuracy every second epoch, but overall performance did not improve much.

**Three or More Convolutional Layers**

Testing networks with 3 to 5 convolutional layers showed that 4 layers yielded the best balance between performance and training time. Adding a repeat() function helped prevent data shortages every second epoch, and data augmentation (by flipping, rotating, and zooming images) enhanced model robustness, increasing the overall accuracy.

For the model with 4 layers, the validation accuracy was 0.30, test accuracy was 0.29, and training accuracy was 0.72. This model showed better performance than the previous ones, but still suffered from overfitting and could not generalize well. While augmentation improved the model, its impact was less than expected.

**Regularization and Dropout**

To address overfitting, I implemented L2 regularization and dropout, but these techniques did not significantly improve the performance. The model failed to learn effectively with these adjustments, as seen in the graph showing little improvement despite changes in dropout values and regularization parameters.
Epochs and Training Time

The networks were trained for 100 epochs, which was a reasonable amount given the size of the dataset. Training for more epochs would have been too time-consuming. For larger image sizes, such as 256x256, I experimented with 40 epochs, as the training time was excessively long for 100 epochs at that resolution.

**Conclusion**

**In conclusion,** the networks showed slow but steady improvement throughout the development process. Although the three networks performed similarly, increasing the complexity of the models resulted in slightly better performance. The accuracy could likely improve with a larger and higher-quality dataset. Given more time and computational resources, further refinement could significantly enhance the model's performance. Additionally, adjusting the training-test-validation split had a slight positive impact on accuracy, but the improvements were marginal.
