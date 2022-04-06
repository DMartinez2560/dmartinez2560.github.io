# Comparing Unsupervised and Supervised Techniques for Instance-level Segmentation: Cityscapes Dataset
### Heriberto A Nieves, Daniel Enrique Martinez, Juan-Diego Florez-Castillo, Kartik Ramachandruni, Vivek Mallampati 

## Introduction

### Problem Definition

Image segmentation is the task of identifying individual objects in the
image based on class, occurrence, and presence in either the foreground
or background of the image. Our project explores instance segmentation,
which combines object detection and semantic segmentation for foreground
objects. Instance segmentation provides the necessary information about
the scene---the class of each object, the frequency of each class, and
pixel-level segmentation masks of objects.

### Background

There are numerous real world application to instance segmentation which
make it an essential yet challenging research problem. For instance, the
biomedical community relies on deep learning techniques to identify
nuclei in electron microscopy images or to locate malignant cancer cells
from CT scans; these act as acting as visual aids to perform more
accurate diagnosis [1, 2].

Scene understanding is not possible without instance segmentation. By
acquiring information about objects in an image, we can perform numerous
computer vision tasks such as image captioning, video analysis and
visual question answering [3].

Autonomous vehicle navigation also makes use of instance segmentation
 [4]. The navigation frameworks need to be able to
distinguish between pedestrians, other vehicles, traffic symbols, and
background objects. The planning algorithms should also further identify
sub-classes of these objects in order to appropriately react to
situations while on the road.

### Related Work

Images of urban environments are often used to test novel deep learning
techniques for segmentation. The Panoptic-DeepLab
algorithm[5], for example, uses decoupled spatial
pyramid pooling layers and dual decoder modules to simultaneously
complete instance and semantic-level segmentation. The FASSST
algorithm[6] performs real-time instance-level
segmentation at video-grade speed by implementing an instance attention
module to efficiently segment target regions and a multi-layer feature
fusion model to obtain regions of interest and class probabilities.
Works have also considered unsupervised learning techniques for image
segmentation. For example, a Local Gaussian Mixture Model (GMM) with an
additional penalty term and a local bias function has been used to
segment noisy images affected by
intensity-nonhomogeneity[7].

### Cityscapes Dataset

We used the Cityscapes dataset for this project [8].
Cityscapes is a collection of on-board vehicle data such as GPS,
odometry, stereo images and disparity maps taken from drives of 27
cities along with annotations of street images using 30 visual classes.
This dataset is abundant in examples with 5000 fine annotations and
20000 coarse annotations of urban street scenes. The dataset has also
been extensively cited by prior work and there are numerous established
benchmarks. We used raw images and the annotation images which contains
rough, pixel-wise labels of objects in the scene.

Clustering algorithms are a group of unsupervised machine learning
techniques which aim to group unlabelled data points according to
similarities in their feature representations. Examples of clustering
algorithms include the K-Means algorithm, GMM, and DBSCAN.

For our project, we implemented the K-Means and DBSCAN algorithms to
perform pixel-level segmentation on raw images from the Cityscapes data
set. To aid these techniques, we employed computer vision techniques to
perform dimensionality reduction as well as extract meaningful feature
representations of the image.

Since the aforementioned algorithms are unsupervised, we are unable to
predict class labels for individual objects in the image. Instead, the
objective of using these methods is to explore how traditional feature
extraction combined with data-efficient clustering techniques performs
on a complex image segmentation data sets such as Cityscapes. Cityscapes
provides both non-sequential and sequential images, the latter of which
presents opportunities for cross-image clustering, image classification
by clustering, and feature extraction.

## Unsupervised Learning - Clustering

### Classical K-Means

The K-Means algorithm, described in
Eq. [1] was implemented on the Cityscapes raw images to
perform pixel-based clustering. Given a set of observations, $X$, we
want to group the observations into `k` sets, `S`, to minimize the
in-cluster variance by operating using the in-cluster mean, $\mu$.

$$\text{argmin}_s \sum_{i=1}^{k} \sum_{x \in S_i} ||x - u_i||^2 = \text{argmin}_s \sum_{i=1}^{k} |S_i|\text{Var}S_i$$

We used the \"RGB\" color-space for clustering and used a resize
operation to reduce the dimensions of the images to be clustered from
1024x2048x3 to a scaled pixel-wdith of 300. The a sample result set of
the K-Means operation is shown in
Fig. [\[fig:kmeans\]](#fig:kmeans){reference-type="ref"
reference="fig:kmeans"}.

![image](figures/cluster_og.png){width="0.9\\columnwidth"}
![image](figures/cluster5.png){width="0.9\\columnwidth"}
![image](figures/cluster10.png){width="0.9\\columnwidth"}

### HOG Feature Extraction

In order to reduce the dimensionality of the raw image data --
Cityscapes image dimensions are 1024x2048x3 -- and extract relevant
feature representations from the images, we chose to generate a
'Histogram of Oriented Gradients', or HOG, descriptor for each image
patch in the image. A HOG descriptor is a histogram of the image
gradient orientation calculated from a localized image patch.

HOGs are popular image descriptors used to match key-points between
images. They are more commonly used to generate features such as
SIFT [9] and SURF [10]. HOG descriptors are also
used in object detection methods by using these image descriptors along
with supervised learning algorithms such as Support Vector Machines to
perform object or human detection [11]. We aim to use
these descriptors as input to unsupervised clustering techniques and
observe the performance on pixel-wise image segmentation.

![image](figures/elbow_plot_aachen.png){width="0.9\\columnwidth"}
[]{#fig:elbow_aachen label="fig:elbow_aachen"}

![image](figures/elbow_plot_nuremburg.png){width="0.9\\columnwidth"}
[]{#fig:elbow_nur label="fig:elbow_nur"}

![image](figures/slh_coeff_aachen.png){width="0.9\\columnwidth"}
[]{#fig:slh_aachen label="fig:slh_aachen"}

![image](figures/slh_coeff_nuremburg.png){width="0.9\\columnwidth"}
[]{#fig:slh_nur label="fig:slh_nur"}

We now discuss the qualitative results obtained from using the K-Means
algorithm.

#### HOG Features with K-Means

For the K-Means clustering method with HOG features, we selected the
data of two cities to evaluate -- Aachen and Nuremberg. For each city,
we selected 10 images to train the clustering algorithm and test on 3
images.

Prior to extracting HOG features, we convert the image to gray-scale and
perform canny edge detection on the image. We observed a significant
improvement in the quality of segmentation with these pre-processing
techniques as well as a reduction in the Sum of Squares value for the
same number of clusters.

To extract the HOG descriptors, we choose image patches of size (16, 16)
with 4 cells in each image patch. We then generate a gradient histogram
of 32 bins for each cell, resulting in a 128-length descriptor for each
image patch. We also resize the image from 1024x2048 to 360x720 to
reduce the computational cost of segmentation.

In order to determine the number of clusters, we first used the elbow
method by plotting the Sum of Squares value calculated from test images
versus the number of clusters. The elbow plots are shown in Fig
[\[fig:elbow\]](#fig:elbow){reference-type="ref" reference="fig:elbow"}.
As the elbow plots did not provide any conclusive value for the ideal
number of clusters, we also plotted the average silhouette coefficient
of test samples for each clusters. The plots of average silhouette
coefficients are shown in Fig
[\[fig:slh\]](#fig:slh){reference-type="ref" reference="fig:slh"}. From
these graphs, we find that the highest silhouette coefficients for the
Aachen and Nuremberg data exist at 10 and 4 clusters respectively.

#### Qualitative Comparison and Discussion

::: figure*
![image](figures/aachen_test_results.png){width="90%"}
:::

::: figure*
![image](figures/nur_test_results.png){width="90%"}
:::

Figures
[\[fig:clusters_aachen\]](#fig:clusters_aachen){reference-type="ref"
reference="fig:clusters_aachen"} and
[\[fig:clusters_nur\]](#fig:clusters_nur){reference-type="ref"
reference="fig:clusters_nur"} show the final segmentation masks obtained
for each of the three test images taken from the Aachen and Nuremburg
data. Looking at these images, we make the following observations:

1.  The segmentation masks are very sensitive to edges and corners in
    the image. This is especially prominent in the Aachen images, where
    surface markings on the road are being separated from the road. This
    is mainly because the gradient of an image is very high at edges,
    corners and other sharp changes in image intensity due to which they
    tend to overpower the image descriptor.

2.  A consequence of the previous point is that cars that appear close
    together are merged with the background, particularly in the
    Nuremburg images. Objects appearing close to each other will not
    have a solid boundary around them. Therefore, the gradient of the
    image at that point is not strong enough for the clustering
    algorithm to separate the two objects from themselves and the
    algorithm instead combines that object with the background
    buildings.

3.  The segmentation masks make an effort to separate objects closer to
    the camera from the background in a few instances. For example, in
    the Nuremburg images, the road and nearby trees are distinctly
    separate from faraway cars and buildings. A possible application of
    this feature could be to use this kind of clustering to perform
    foreground extraction in images.

To summarize, we believe that performing unsupervised clustering either
directly on the raw RGB images or on image descriptors, like HOG, is
insufficient to perform meaningful pixel-level segmentation. To extend
this approach, other low-dimensional feature representations of images
need to be explored that can capture both local variations in pixel
intensities as well as global information about which pixels belong to
which object.

### DBSCAN

Density-based spatial clustering of applications with noise (DBSCAN) is
a clustering method that groups together closely packed points. It
divides points into three categories: core points, border points, and
outliers. The clustering is based on two main parameters: Eps and
MinPoints. Eps is the distance from a point for which the algorithm
looks for nearby points to evaluate the density and MinPoints is the
threshold for the number of points in the range defined by Eps necessary
to mark a point as high or low density. Core points have more than the
defined number of points in their neighborhood. Border points have less
points than defined in their neighborhood but are in the neighborhood of
a core point. Outliers are points that aren't core points or border
points. Core points serve as the interior of clusters and border points
as the edges of clusters.

In Figure [\[fig:dbs_aach\]](#fig:dbs_aach){reference-type="ref"
reference="fig:dbs_aach"} we show the results of DBSCAN clustering on 4
images from the Aachen data set. The clustering seems to separate the
road in the images from the background objects. This is likely due to
the large size of the road in the frame and it's uniformity in color
making it easy to segment. The background has a lot of features and
varying objects which all get grouped together as high density. Where
the road is more clearly defined by borders and medians, the edges of
the cluster of the road is preserved very well. In the images with more
objects around the edges of the road, the division is not as clear.

We plan on comparing the performance of DBSCAN on the raw images against
using pre-processing techniques such as extracting the HOG features.

::: figure*
![](figures/DBScan_aachen.JPG){width="90%"}
:::

## Supervised Learning - Model Chaining

Supervised algorithms were performed using the fine annotations of the
Cityscapes dataset. Two methods of completing pixel-level segmentation
were completed. In this case, multiple objects are within the images.
The first approach entails identifying the objects and finding the
corresponding pixel for the object via model chaining. The second and
more direct approach involves training on the pixel information and
obtaining the pixel level map using deep neural network architecture.

### Methods

**Method 1: Model chaining** involved using Google Colab to input raw,
unlabeled (n=10) Cityscapes images,
[\[fig:DETR4\]](#fig:DETR4){reference-type="ref"
reference="fig:DETR4"}A, into the DEtection TRansformer (DETR) model
(pre-trained on the COCO dataset) [12]. Each of the images
were from the Nuremburg dataset. Primary outputs of the model include
images with bounding boxes labeled,
[\[fig:DETR4\]](#fig:DETR4){reference-type="ref"
reference="fig:DETR4"}B, and a list of bounding boxes. In the future,
the model outputs will then be inputted into the Segment My Object
(SegMyO) pipeline which takes bounding box information and applies masks
to the raw image using Mask R-CNN [13]. DETR models
were compared for performance and computation time to best select a
model for use with SegMyO. Predicted outputs of the entire system are
final pixel-level labeled images that can be evaluated against the
Cityscapes fine annotated data.

::: figure*
![image](figures/Image4Results.pdf){width="90%"}
:::

**Method 2: Deep Neural Network** utilizes the annotation masks to train
the model uses the architectures like RESNET50 or VGG-16 [14] to
create the masks for the object instances. The latest development of
algorithms in CNN and DL pushed to create faster and lighter
(computational) networks which create segmentation masks. We tested the
pre-trained neural network models of DeepLab on the Cityscapes dataset
and compare the pixel-level segmentation performances between various
architectures[15]. One objective was to identify any
performance advantages when using the transformers combined with neural
networks. The models DeepLab[14] was implemented as an end to
end pre-trained neural network on Cityscapes to compare different
architectures and seeing qualitative measure comparisons.

### Model Architecture Descriptions

::: enumerate
1.  **DETR**: It approaches object detection as a direct set prediction
    problem. It consists of a set-based global loss, which forces unique
    predictions via bipartite matching, and a Transformer
    encoder-decoder architecture. Given a fixed small set of learned
    object queries, DETR reasons about the relations of the objects and
    the global image context to directly output the final set of
    predictions in parallel. Due to this parallel nature, DETR is very
    fast and efficient[16].

2.  **SegMyO**: It automatically extracts the segmented objects in
    images based on given bounding boxes. When provided with the
    bounding box, it looks for the output object with the best coverage,
    based on several geometric criteria. Associated with a semantic
    segmentation model trained on a similar dataset(PASCAL-VOC and COCO)
    this pipeline provides a simple solution to segment efficiently a
    dataset without requiring specific training, but also to the problem
    of weakly-supervised segmentation. This is particularly useful to
    segment public datasets available with weak object annotations
    coming from an algorithm (in our case DETR).

3.  **Panoptic-DeepLab**: It is a state-of-the-art box-free system for
    panoptic segmentation, where the goal is to assign a unique value,
    encoding both semantic label and instance ID, to every pixel in an
    image. The class-agnostic instance segmentation is first obtained by
    grouping the predicted foreground pixels (inferred by semantic
    segmentation) to their closest predicted instance centers. To
    generate final panoptic segmentation, we then fuse the
    class-agnostic instance segmentation with semantic segmentation by
    the efficient majority-vote scheme.[15]

4.  **Axial-DeepLab**: It incorporates the powerful axial self-attention
    modules, also known as the encoder of Axial Transformers, for
    general dense prediction tasks. The backbone of Axial-DeepLab,
    called Axial-ResNet, is obtained by replacing the residual blocks in
    any type of ResNets with our proposed axial-attention blocks. They
    adopt the hybrid CNN-Transformer architecture, where they stack the
    effective axial-attention blocks on top of the first few stages of
    ResNets. This hybrid CNN-Transformer architecture is very effective
    on segmentation tasks[17].
:::

::: figure*
![image](figures/deepLab_combine_images.png){width="90%"}
:::

### Quantitative Metrics

::: enumerate
1.  **Mean Intersection over Union (mIoU)** is another method to
    evaluate the predictions from an image segmentation model. This is a
    metric that takes the IoU over all of the classes and takes the mean
    of them. This is a good indicator of how well an image segmentation
    model performs over all the classes that the model would want to
    detect.

2.  **Computational Time** is an important metric of evaluation. The age
    of cloud computing, the focus of researchers are moving towards
    accuracy, but in reality compute is still scarce and is a matter of
    concern. Hence we will also be looking into determining time
    efficient methods by using the inbuilt tools to calculate.

3.  **Object count** is a metric of evaluation for the object
    recognition algorithm used in Method 1. The simple metric can help
    us keep on track and measure the performance of the algorithm.
:::

### Results

Definitions of the metrics used to report results in tables and images
include:

1.  Total Run Time refers to the duration of the code per image
    iteration.

2.  Model Run Time refers to the duration of running the model per image
    iteration.

3.  Mean Score refers to the score per image iteration

4.  Mean Score Per Car refers to the score of all the cars per image
    iteration.

5.  Cars Labeled refers to the number of cars labeled in the image

6.  Items Labeled refers to the number of items labeled in the image.

Fig [\[fig:deeplab_results\]](#fig:deeplab_results){reference-type="ref"
reference="fig:deeplab_results"} show that DeepLab identifies all the
semantic classes and all instances(additionally). Table
[4](#table:deepLabmIoU){reference-type="ref"
reference="table:deepLabmIoU"} shows that both Panoptic and Axial
algorithms give an mIoU greater than 0.9 for Aachen dataset. Few of the
images perform slightly better for Panoptic and others for Axial. We
need to do more extensive testing on the total validation dataset to
give a greater sense of impact of algorithms on data. Few features like
small objects can be better recognized in few architecture than others.

Fig [\[fig:DETR4\]](#fig:DETR4){reference-type="ref"
reference="fig:DETR4"}, [\[fig:DETR8\]](#fig:DETR8){reference-type="ref"
reference="fig:DETR8"}, and
[\[fig:DETR10\]](#fig:DETR10){reference-type="ref"
reference="fig:DETR10"} help us qualitatively understand the impact of
the DETR algorithm. The bounding boxes help recognize the objects in the
Cityscapes dataset. This is particularly interesting as DETR was trained
on COCO dataset and has not been trained on Cityscapes data before
running it on validation. Only items identified by the DETR algorithms
with confidence scores of 0.9 or above were kept.

The tables [1](#table:Runtime){reference-type="ref"
reference="table:Runtime"}, [2](#table:scores){reference-type="ref"
reference="table:scores"}, and
[3](#table:itemCount){reference-type="ref" reference="table:itemCount"}
help us see which RESNET model is better and the comparison of the
architecture helps us better to make the network better with latest
additions. A relatively constant duration of about 0.7 seconds was found
across each DETR model based on the difference between Total Time and
Model Time in [1](#table:Runtime){reference-type="ref"
reference="table:Runtime"}.

::: center
::: {#table:Runtime}
       Model        Total Run Time   Model Run Time
  ---------------- ---------------- ----------------
     Resnet 101         9.459            8.756
   Resnet 101-DC5       20.445           19.744
   Resnet 50-DC5        18.169           17.467
     Resnet 50          6.359            5.655

  : Mean Run time per Image Iteration for Each DETR Architecture
:::

[]{#table:Runtime label="table:Runtime"}
:::

::: center
::: {#table:scores}
       Model        Mean Score   Mean Score Per Car
  ---------------- ------------ --------------------
     Resnet 101       0.971            0.973
   Resnet 101-DC5     0.975            0.977
   Resnet 50-DC5      0.972            0.975
     Resnet 50        0.972            0.974

  : Mean Confidence Scores by Each DETR Architecture
:::

[]{#table:scores label="table:scores"}
:::

::: center
::: {#table:itemCount}
       Model        Cars Labeled   Items Labeled
  ---------------- -------------- ---------------
     Resnet 101         126             185
   Resnet 101-DC5       133             190
   Resnet 50-DC5        134             202
     Resnet 50          121             184

  : Number of Objects Labeled by Each DETR Architecture
:::

[]{#table:itemCount label="table:itemCount"}
:::

::: center
::: {#table:deepLabmIoU}
   Images    Panoptic-DeepLab   Axial-DeepLab
  --------- ------------------ ---------------
   Image 1        0.9364           0.9393
   Image 2        0.9348           0.9307
   Image 3        0.9271           0.9208
   Image 4        0.9133           0.9240

  : mIoU for DeepLab Models
:::

[]{#table:deepLabmIoU label="table:deepLabmIoU"}
:::

::: figure*
![image](figures/Image10Results.pdf){width="90%"}
:::

::: figure*
![image](figures/Image8Results.pdf){width="90%"}
:::

### Discussion

We observed that the DETR performance for smaller items increases with
the DC-5 models as the resolution is increased by a factor of 2,
correlating with increases in Items Labeled in Table
[3](#table:itemCount){reference-type="ref" reference="table:itemCount"}.
On the down side, the DC-5 models have a higher computation cost because
of the higher cost associated with self-attentions of the encoders, as
shown in Table [1](#table:Runtime){reference-type="ref"
reference="table:Runtime"}. This observation matches with the claim made
in the DETR paper[16].

Taking the unorthodox way, we are trying to combine two non-related
models and without linking them, but rather having the output of one be
processed into being input for the other algorithm. We are yet to
experiment with the full Model Chaining approach and identify advantages
of the system. We are optimistic because the DETR output is bounding
boxes of objects in an image, and the input for segmentation for SegMyO
are also bounding boxes. We need to get object wise mask, and take a
union of the masks to create a bigger image mask. We can evaluate with
the ground truth masks once the total image mask is created.

Future work will involve comparing the DETR models against each other
image by image using all the metrics and also further qualitative
analysis. This will aid in determining missed or incorrect labels. Due
to its high score, relative items labeled count, and low computation
time, the Resnet 50 model appears favorable for chaining with SegMyO.
With that said, a final decision will be made upon the completion of the
further analyses.

DeepLab results (Figure
[\[fig:deeplab_results\]](#fig:deeplab_results){reference-type="ref"
reference="fig:deeplab_results"}) clearly shows that the model is
successful qualitative and the (Table
[4](#table:deepLabmIoU){reference-type="ref"
reference="table:deepLabmIoU"}) shows the model is successful
quantitatively on average. The discussion for the DeepLab is between
using different architectures. The RESNET-50 being backbone is a rather
famous architecture, but we also extend our implementation to use the
axial-transformers combined with ResNets. We are planning to run the
model on the total validation dataset and complete the analysis on the
DeepLab models trained on Cityscapes.

## References
1. O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional
networks for biomedical image segmentation,” in International Confer-
ence on Medical image computing and computer-assisted intervention.
Springer, 2015, pp. 234–241.
2. D. Mzurikwao, M. U. Khan, O. W. Samuel, J. Cinatl, M. Wass,
M. Michaelis, G. Marcelli, and C. S. Ang, “Towards image-based
cancer cell lines authentication using deep neural networks,” Scientific
reports, vol. 10, no. 1, pp. 1–15, 2020.
3. W. Cai, Z. Xiong, X. Sun, P. L. Rosin, L. Jin, and X. Peng, “Panoptic
segmentation-based attention for image captioning,” Applied Sciences,
vol. 10, no. 1, p. 391, 2020.
4. D. de Geus, P. Meletis, and G. Dubbelman, “Single network panoptic
segmentation for street scene understanding,” in 2019 IEEE Intelligent
Vehicles Symposium (IV). IEEE, 2019, pp. 709–715.
5. B. Cheng, M. D. Collins, Y. Zhu, T. Liu, T. S. Huang, H. Adam, and
L.-C. Chen, “Panoptic-deeplab: A simple, strong, and fast baseline for
bottom-up panoptic segmentation,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2020, pp.
12 475–12 485.
6. Y. Cheng, R. Lin, P. Zhen, T. Hou, C. W. Ng, H.-B. Chen, H. Yu, and
N. Wong, “Fassst: Fast attention based single-stage segmentation net
for real-time instance segmentation,” in Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision, 2022, pp.
2210–2218.
7. J. Liu and H. Zhang, “Image segmentation using a local gmm in a
variational framework,” Journal of mathematical imaging and vision,
vol. 46, no. 2, pp. 161–176, 2013.
8. M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Be-
nenson, U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset
for semantic urban scene understanding,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2016, pp.
3213–3223.
9. D. G. Lowe, “Object recognition from local scale-invariant features,”
in Proceedings of the seventh IEEE international conference on
computer vision, vol. 2. Ieee, 1999, pp. 1150–1157.
10. H. Bay, T. Tuytelaars, and L. V. Gool, “Surf: Speeded up robust
features,” in European conference on computer vision. Springer, 2006,
pp. 404–417.
11. N. Dalal and B. Triggs, “Histograms of oriented gradients for human
detection,” in 2005 IEEE computer society conference on computer
vision and pattern recognition (CVPR’05), vol. 1. Ieee, 2005, pp.
886–893.
12. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and
S. Zagoruyko, “End-to-end object detection with transformers,” in
European conference on computer vision. Springer, 2020, pp. 213–
229.
13. R. Del ́earde, C. Kurtz, P. Dejean, and L. Wendling, “Segment my
object: A pipeline to extract segmented objects in images based on
labels or bounding boxes.” in VISIGRAPP (5: VISAPP), 2021, pp.
618–625.
14. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille,
“Deeplab: Semantic image segmentation with deep convolutional
nets, atrous convolution, and fully connected crfs,” 2016. [Online].
Available: https://arxiv.org/abs/1606.00915
15. B. Cheng, M. D. Collins, Y. Zhu, T. Liu, T. S. Huang, H. Adam, and
L.-C. Chen, “Panoptic-DeepLab: A simple, strong, and fast baseline
for bottom-up panoptic segmentation,” in CVPR, 2020.
16. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and
S. Zagoruyko, “End-to-end object detection with transformers,” 2020.
[Online]. Available: https://arxiv.org/abs/2005.12872
17. H. Wang, Y. Zhu, B. Green, H. Adam, A. Yuille, and L.-C. Chen,
“Axial-DeepLab: Stand-alone axial-attention for panoptic segmenta-
tion,” in ECCV, 2020.

## References

1. O. Ronneberger, P. Fischer, and T. Brox, “U-net: Convolutional
networks for biomedical image segmentation,” in International Confer-
ence on Medical image computing and computer-assisted intervention.
Springer, 2015, pp. 234–241.
2. D. Mzurikwao, M. U. Khan, O. W. Samuel, J. Cinatl, M. Wass,
M. Michaelis, G. Marcelli, and C. S. Ang, “Towards image-based
cancer cell lines authentication using deep neural networks,” Scientific
reports, vol. 10, no. 1, pp. 1–15, 2020.
3. W. Cai, Z. Xiong, X. Sun, P. L. Rosin, L. Jin, and X. Peng, “Panoptic
segmentation-based attention for image captioning,” Applied Sciences,
vol. 10, no. 1, p. 391, 2020.
4. D. de Geus, P. Meletis, and G. Dubbelman, “Single network panoptic
segmentation for street scene understanding,” in 2019 IEEE Intelligent
Vehicles Symposium (IV). IEEE, 2019, pp. 709–715
5. B. Cheng, M. D. Collins, Y. Zhu, T. Liu, T. S. Huang, H. Adam, and
L.-C. Chen, “Panoptic-deeplab: A simple, strong, and fast baseline for
bottom-up panoptic segmentation,” in Proceedings of the IEEE/CVF
conference on computer vision and pattern recognition, 2020, pp.
12 475–12 485.
6. Y. Cheng, R. Lin, P. Zhen, T. Hou, C. W. Ng, H.-B. Chen, H. Yu, and
N. Wong, “Fassst: Fast attention based single-stage segmentation net
for real-time instance segmentation,” in Proceedings of the IEEE/CVF
Winter Conference on Applications of Computer Vision, 2022, pp.
2210–2218.
7. J. Liu and H. Zhang, “Image segmentation using a local gmm in a
variational framework,” Journal of mathematical imaging and vision,
vol. 46, no. 2, pp. 161–176, 2013.
8. M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, 
U. Franke, S. Roth, and B. Schiele, “The cityscapes dataset
for semantic urban scene understanding,” in Proceedings of the IEEE
conference on computer vision and pattern recognition, 2016, pp.
3213–3223.
9. D. G. Lowe, “Object recognition from local scale-invariant features,”
in Proceedings of the seventh IEEE international conference on
computer vision, vol. 2. Ieee, 1999, pp. 1150–1157.
10. H. Bay, T. Tuytelaars, and L. V. Gool, “Surf: Speeded up robust
features,” in European conference on computer vision. Springer, 2006,
pp. 404–417.
11. N. Dalal and B. Triggs, “Histograms of oriented gradients for human
detection,” in 2005 IEEE computer society conference on computer
vision and pattern recognition (CVPR’05), vol. 1. Ieee, 2005, pp.
886–893.
12. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and
S. Zagoruyko, “End-to-end object detection with transformers,” in
European conference on computer vision. Springer, 2020, pp. 213–
229.
13. R. Delearde, C. Kurtz, P. Dejean, and L. Wendling, “Segment my ´
object: A pipeline to extract segmented objects in images based on
labels or bounding boxes.” in VISIGRAPP (5: VISAPP), 2021, pp.
618–625.
14. L.-C. Chen, G. Papandreou, I. Kokkinos, K. Murphy, and A. L. Yuille,
“Deeplab: Semantic image segmentation with deep convolutional
nets, atrous convolution, and fully connected crfs,” 2016. [Online].
Available: https://arxiv.org/abs/1606.00915
15. B. Cheng, M. D. Collins, Y. Zhu, T. Liu, T. S. Huang, H. Adam, and
L.-C. Chen, “Panoptic-DeepLab: A simple, strong, and fast baseline
for bottom-up panoptic segmentation,” in CVPR, 2020.
16. N. Carion, F. Massa, G. Synnaeve, N. Usunier, A. Kirillov, and
S. Zagoruyko, “End-to-end object detection with transformers,” 2020.
[Online]. Available: https://arxiv.org/abs/2005.12872
17. H. Wang, Y. Zhu, B. Green, H. Adam, A. Yuille, and L.-C. Chen,
“Axial-DeepLab: Stand-alone axial-attention for panoptic segmentation,” 
in ECCV, 2020.
