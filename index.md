## Comparing Unsupervised and Supervised Techniques for Instance-level Segmentation: Cityscapes Dataset
### Heriberto A Nieves, Daniel Enrique Martinez, Juan Diego Florez-Castillo, Kartik Ramachandruni, Vivek Mallampati 

#### Introduction
Image segmentation is the task of identifying individual objects in the image based on class, occurrence, and whether they lie in the foreground or background. Our project explores instance segmentation, which combines object detection and semantic segmentation for foreground objects. Instance segmentation provides us with the necessary information about the scene – the class of each object, the frequency of each class, and pixel-level segmentation masks of objects—the diverse range of information that instance segmentation outputs, thus making the task a challenging problem-solving.  

Numerous real-world applications to instance segmentation make it a critical research problem. For example, the biomedical community relies on deep learning techniques to identify distinct types of cells in electron microscopy images<sup>[1]</sup> or CT scans of cancer patients<sup>[2]</sup>. Instance segmentation is also a crucial component for scene understanding for applications such as image captioning<sup>[3]</sup>. In addition, segmentation techniques help autonomous navigation services distinguish between pedestrians, vehicles, traffic symbols, and background objects and identify subclasses of these objects<sup>[4]</sup>. Therefore, instance segmentation is a crucial component of many real-world vision systems. 

Images of urban environments have often been the targeted datasets to test novel deep learning techniques, for instance, segmentation. The Panoptic-DeepLab algorithm<sup>[5]</sup>, for example, uses de-coupled spatial pyramid pooling layers and dual decoder modules to simultaneously complete instance and semantic-level segmentation. The FASSST algorithm<sup>[6]</sup> performs real-time instance-level segmentation at video-grade speed by implementing an instance attention module to efficiently segment target regions and a multi-layer feature fusion model to obtain regions of interest and class probabilities. Works have also considered unsupervised learning techniques for image segmentation – for example, a Local GMM with an additional penalty term and a local bias function to segment noisy images affected by intensity inhomogeneity<sup>[7]</sup>. 
 
#### Methods
For these reasons, we have chosen the Cityscapes dataset<sup>[8]</sup> for this project. Cityscapes is a collection of onboard vehicle data taken from drives of 27 cities with annotations of street images using 30 visual classes. This dataset is abundant in examples with 5000 fine annotations and 20000 coarse annotations of urban street scenes. We propose implementing the following solutions: 

1.	Density-based Clustering: We will explore the performance of this clustering by comparing the following image transformation techniques: a) foreground extraction, b) use of image features such as SIFT, c) preprocessing such as edge detection and color thresholding. 
2.	Unsupervised domain adaptation: The unsupervised domain-adaptive GAN-method CCyC-PDAM<sup>[9]</sup> was successfully used for microscopy image analysis. We will tune and retrain this architecture for Cityscapes. 
3.	Supervised model chaining: We will combine the pre-trained object detection capabilities of DETR<sup>[10]</sup> with the segmentation generation of SegMyO<sup>[11]</sup> and test on Cityscapes. 
4.	Supervised Domain adaptation: We will use transfer learning to improve the performance of the PolyTransform architecture<sup>[12]</sup> (trained on COCO dataset) on Cityscapes. 

#### Results and Discussion
The contribution of our project is an extensive quantitative and qualitative comparison of the methods. We anticipate that supervised models will outperform unsupervised models on quantitative metrics. However, we are mainly interested in studying the qualitative differences in these models.

### Proposal Video

[![Project Proposal Video](https://i.imgur.com/nwkfvJw.png)](https://youtu.be/svXH5WHxNUU "Project Proposal Video - Click to Watch!")


### Project Timeline:
![Project Timeline Gantt Chart](/assets/images/Gantt.PNG)

### References

1.	Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." International Conference on Medical image computing and computer-assisted intervention. Springer, Cham, 2015.
2.	Mzurikwao, Deogratias, et al. "Towards image-based cancer cell lines authentication using deep neural networks." Scientific reports 10.1 (2020): 1-15.
3.	Cai, Wenjie, et al. "Panoptic segmentation-based attention for image captioning." Applied Sciences 10.1 (2020): 391.
4.	D. de Geus, P. Meletis and G. Dubbelman, "Single Network Panoptic Segmentation for Street Scene Understanding," 2019 IEEE Intelligent Vehicles Symposium (IV), 2019, pp. 709-715, doi: 10.1109/IVS.2019.8813788.
5.	Cheng, Bowen et al. “Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation.” 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2020): 12472-12482.
6.	Y. Cheng et al., "FASSST: Fast Attention Based Single-Stage Segmentation Net for Real-Time Instance Segmentation," 2022 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 2022, pp. 2714-2722, doi: 10.1109/WACV51458.2022.00277.
7. Liu, J., Zhang, H. Image Segmentation Using a Local GMM in a Variational Framework. J Math Imaging Vis 46, 161–176 (2013). https://doi.org/10.1007/s10851-012-0376-5
9.	M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele, “The Cityscapes Dataset for Semantic Urban Scene Understanding,” in Proc. of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016. [Paper]
10.	Liu, Dongnan, et al. "Pdam: A panoptic-level feature alignment framework for unsupervised domain adaptive instance segmentation in microscopy images." IEEE Transactions on Medical Imaging 40.1 (2020): 154-165.
11.	Carion, Nicolas, et al. "End-to-end object detection with transformers." European conference on computer vision. Springer, Cham, 2020.
12.	Deléarde, Robin, et al. "Segment My Object: A Pipeline to Extract Segmented Objects in Images based on Labels or Bounding Boxes." VISIGRAPP (5: VISAPP). 2021.
13.	Liang, Justin, et al. "Polytransform: Deep polygon transformer for instance segmentation." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.


