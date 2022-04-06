## Comparing Unsupervised and Supervised Techniques for Instance-level Segmentation: Cityscapes Dataset
### Heriberto A Nieves, Daniel Enrique Martinez, Juan Diego Florez-Castillo, Kartik Ramachandruni, Vivek Mallampati 

Image segmentation is the task of identifying individual objects in the image based on class, occurrence, and presence in either the foreground or background of the image. Our project explores instance segmentation, which combines object detection and semantic segmentation for foreground objects. Instance segmentation provides the necessary information about the scene---the class of each object, the frequency of each class, and pixel-level segmentation masks of objects.


#### Background
There are numerous real world application to instance segmentation which make it an essential yet challenging research problem.
For instance, the biomedical community relies on deep learning techniques to identify nuclei in electron microscopy images or to locate malignant cancer cells from CT scans; these act as acting as visual aids to perform more accurate diagnosis~\cite{ronneberger2015u, mzurikwao2020towards}. 

Scene understanding is not possible without instance segmentation. By acquiring information about objects in an image, we can perform numerous computer vision tasks such as image captioning, video analysis and visual question answering~\cite{cai2020panoptic}.

Autonomous vehicle navigation also makes use of instance segmentation
~\cite{de2019single}. The navigation frameworks need to be able to distinguish between pedestrians, other vehicles, traffic symbols, and background objects. The planning algorithms should also further identify sub-classes of these objects in order to appropriately react to situations while on the road.


#### Related Work
Images of urban environments are often used to test novel deep learning techniques for segmentation. The Panoptic-DeepLab algorithm\cite{cheng2020panoptic}, for example, uses decoupled spatial pyramid pooling layers and dual decoder modules to simultaneously complete instance and semantic-level segmentation. The FASSST algorithm\cite{cheng2022fassst} performs real-time instance-level segmentation at video-grade speed by implementing an instance attention module to efficiently segment target regions and a multi-layer feature fusion model to obtain regions of interest and class probabilities. Works have also considered unsupervised learning techniques for image segmentation. For example, a Local Gaussian Mixture Model (GMM) with an additional penalty term and a local bias function has been used to segment noisy images affected by intensity-nonhomogeneity\cite{liu2013image}.

#### CityScapes Dataset
We used the Cityscapes dataset for this project~\cite{cordts2016cityscapes}. Cityscapes is a collection of on-board vehicle data such as GPS, odometry, stereo images and disparity maps taken from drives of 27 cities along with annotations of street images using 30 visual classes. This dataset is abundant in examples with 5000 fine annotations and 20000 coarse annotations of urban street scenes. The dataset has also been extensively cited by prior work and there are numerous established benchmarks. We used raw images and the annotation images which contains rough, pixel-wise labels of objects in the scene.

Clustering algorithms are a group of unsupervised machine learning techniques which aim to group unlabelled data points according to similarities in their feature representations. Examples of clustering algorithms include the K-Means algorithm, GMM, and DBSCAN.

For our project, we implemented the K-Means and DBSCAN algorithms to perform pixel-level segmentation on raw images from the Cityscapes data set. To aid these techniques, we employed computer vision techniques to perform dimensionality reduction as well as extract meaningful feature representations of the image.

Since the aforementioned algorithms are unsupervised, we are unable to predict class labels for individual objects in the image. Instead, the objective of using these methods is to explore how traditional feature extraction combined with data-efficient clustering techniques performs on a complex image segmentation data sets such as Cityscapes. Cityscapes provides both non-sequential and sequential images, the latter of which presents opportunities for cross-image clustering, image classification by clustering, and feature extraction.

## Unsupervised Methods
### Classical K-Means
The K-Means algorithm, described in Eq.~\eqref{eq:kmeans} was implemented on the Cityscapes raw images to perform pixel-based clustering. Given a set of observations, $X$, we want to group the observations into $k$ sets, $S$, to minimize the in-cluster variance by operating using the in-cluster mean, $\mu$.

\begin{equation}\label{eq:kmeans}
    \text{argmin}_s \sum_{i=1}^{k} \sum_{x \in S_i} \abs{\abs{x - u_i}}^2 = \text{argmin}_s \sum_{i=1}^{k} \abs{S_i}\text{Var}S_i
\end{equation}

 We used the "RGB" color-space for clustering and used a resize operation to reduce the dimensions of the images to be clustered from 1024x2048x3 to a scaled pixel-wdith of 300. The a sample result set of the K-Means operation is shown in Fig.~\ref{fig:kmeans}.

\begin{figure}
    \centering
    \includegraphics[width=0.9\columnwidth]{figures/cluster_og.png}
    \includegraphics[width=0.9\columnwidth]{figures/cluster5.png}
    \includegraphics[width=0.9\columnwidth]{figures/cluster10.png}
    \caption{The K-Means clustering output, where the top image is the original Cityscapes image, the middle image is at 5 clusters, and the bottom image is at 10 clusters.}
    \label{fig:kmeans}
\end{figure}

### HOG Feature Extraction

In order to reduce the dimensionality of the raw image data -- Cityscapes image dimensions are 1024x2048x3 -- and extract relevant feature representations from the images, we chose to generate a 'Histogram of Oriented Gradients', or HOG, descriptor for each image patch in the image. A HOG descriptor is a histogram of the image gradient orientation calculated from a localized image patch.

HOGs are popular image descriptors used to match key-points between images. They are more commonly used to generate features such as SIFT~\cite{lowe1999object} and SURF~\cite{bay2006surf}. HOG descriptors are also used in object detection methods by using these image descriptors along with supervised learning algorithms such as Support Vector Machines to perform object or human detection~\cite{dalal2005histograms}. We aim to use these descriptors as input to unsupervised clustering techniques and observe the performance on pixel-wise image segmentation.

### Qualitative Results

\begin{figure}
    \centering
    \begin{subfigure}
        \centering
        \includegraphics[width=0.9\columnwidth]{figures/elbow_plot_aachen.png}
        \label{fig:elbow_aachen}
    \end{subfigure}
    \begin{subfigure}
        \centering
        \includegraphics[width=0.9\columnwidth]{figures/elbow_plot_nuremburg.png}
        \label{fig:elbow_nur}
    \end{subfigure}
\caption{Elbow plots of the Aachen and Nuremburg test images to determine the optimal number of clusters for the K-Means algorithm.}
\label{fig:elbow}
\end{figure}

\begin{figure}
    \centering
    \begin{subfigure}
        \centering
        \includegraphics[width=0.9\columnwidth]{figures/slh_coeff_aachen.png}
        \label{fig:slh_aachen}
    \end{subfigure}
    \begin{subfigure}
        \centering
        \includegraphics[width=0.9\columnwidth]{figures/slh_coeff_nuremburg.png}
        \label{fig:slh_nur}
    \end{subfigure}
\caption{Silhouette Coefficient plots of the Aachen and Nuremburg test images to determine the optimal number of clusters for the K-Means algorithm. Higher Silhoutte Coefficient indicates better clustering performance.}
\label{fig:slh}
\end{figure}


% \begin{figure}
% \centering
% \subfloat[Aachen Test Images]{%
%   \includegraphics[width=0.9\columnwidth]{figures/elbow_plot_aachen.png}%
% }
% \subfloat[Nuremburg Test Images]{%
%   \includegraphics[width=0.9\columnwidth]{figures/elbow_plot_nuremburg.png}%
% }
% \caption{Elbow plots of the Aachen and Nuremburg test images to determine the optimal number of clusters for the K-Means algorithm.}
% \end{figure}

% \begin{figure}
% \centering
% \subfloat[Aachen Test Images]{%
%   \includegraphics[width=0.9\columnwidth]{figures/slh_coeff_aachen.png}%
% }
% \subfloat[Nuremburg Test Images]{%
%   \includegraphics[width=0.9\columnwidth]{figures/slh_coeff_nuremburg.png}%
% }
% \caption{Silhouette Coefficient plots of the Aachen and Nuremburg test images to determine the optimal number of clusters for the K-Means algorithm. Higher Silhoutte Coefficient indicates better clustering performance.}
% \end{figure}


We now discuss the qualitative results obtained from using the K-Means algorithm. 

### HOG Features with K-Means

For the K-Means clustering method with HOG features, we selected the data of two cities to evaluate -- Aachen and Nuremberg. For each city, we selected 10 images to train the clustering algorithm and test on 3 images. 

Prior to extracting HOG features, we convert the image to gray-scale and perform canny edge detection on the image. We observed a significant improvement in the quality of segmentation with these pre-processing techniques as well as a reduction in the Sum of Squares value for the same number of clusters.

To extract the HOG descriptors, we choose image patches of size (16, 16) with 4 cells in each image patch. We then generate a gradient histogram of 32 bins for each cell, resulting in a 128-length descriptor for each image patch. We also resize the image from 1024x2048 to 360x720 to reduce the computational cost of segmentation.

In order to determine the number of clusters, we first used the elbow method by plotting the Sum of Squares value calculated from test images versus the number of clusters. The elbow plots are shown in Fig \ref{fig:elbow}. As the elbow plots did not provide any conclusive value for the ideal number of clusters, we also plotted the average silhouette coefficient of test samples for each clusters. The plots of average silhouette coefficients are shown in Fig \ref{fig:slh}. From these graphs, we find that the highest silhouette coefficients for the Aachen and Nuremberg data exist at 10 and 4 clusters respectively. 

### Qualitative Comparison and Discussion

\begin{figure*}
\centering
\includegraphics[width=0.9\textwidth]{figures/aachen_test_results.png}
\caption{Test images (left column) from the Aachen data and corresponding segmentation masks (right column) using K-Means with 10 clusters. Each color in the segmentation mask represents a different cluster.}
\label{fig:clusters_aachen}
\end{figure*}

\begin{figure*}
\centering
\includegraphics[width=0.9\textwidth]{figures/nur_test_results.png}
\caption{Test images (left column) from the Nuremburg data and corresponding segmentation masks (right column) using K-Means with 4 clusters. Each color in the segmentation mask represents a different cluster.}
\label{fig:clusters_nur}
\end{figure*}

Figures \ref{fig:clusters_aachen} and \ref{fig:clusters_nur} show the final segmentation masks obtained for each of the three test images taken from the Aachen and Nuremburg data. Looking at these images, we make the following observations:

\begin{enumerate}
    \item The segmentation masks are very sensitive to edges and corners in the image. This is especially prominent in the Aachen images, where surface markings on the road are being separated from the road. This is mainly because the gradient of an image is very high at edges, corners and other sharp changes in image intensity due to which they tend to overpower the image descriptor.
    
    \item A consequence of the previous point is that cars that appear close together are merged with the background, particularly in the Nuremburg images. Objects appearing close to each other will not have a solid boundary around them. Therefore, the gradient of the image at that point is not strong enough for the clustering algorithm to separate the two objects from themselves and the algorithm instead combines that object with the background buildings.
    
    \item The segmentation masks make an effort to separate objects closer to the camera from the background in a few instances. For example, in the Nuremburg images, the road and nearby trees are distinctly separate from faraway cars and buildings. A possible application of this feature could be to use this kind of clustering to perform foreground extraction in images.
\end{enumerate}

To summarize, we believe that performing unsupervised clustering either directly on the raw RGB images or on image descriptors, like HOG, is insufficient to perform meaningful pixel-level segmentation. To extend this approach, other low-dimensional feature representations of images need to be explored that can capture both local variations in pixel intensities as well as global information about which pixels belong to which object. 

### DBSCAN

Density-based spatial clustering of applications with noise (DBSCAN) is a clustering method that groups together closely packed points. It divides points into three categories: core points, border points, and outliers. The clustering is based on two main parameters: Eps and MinPoints. Eps is the distance from a point for which the algorithm looks for nearby points to evaluate the density and MinPoints is the threshold for the number of points in the range defined by Eps necessary to mark a point as high or low density. Core points have more than the defined number of points in their neighborhood. Border points have less points than defined in their neighborhood but are in the neighborhood of a core point. Outliers are points that aren't core points or border points. Core points serve as the interior of clusters and border points as the edges of clusters.

In Figure \ref{fig:dbs_aach} we show the results of DBSCAN clustering on 4 images from the Aachen data set. The clustering seems to separate the road in the images from the background objects. This is likely due to the large size of the road in the frame and it's uniformity in color making it easy to segment. The background has a lot of features and varying objects which all get grouped together as high density. Where the road is more clearly defined by borders and medians, the edges of the cluster of the road is preserved very well. In the images with more objects around the edges of the road, the division is not as clear. 

We plan on comparing the performance of DBSCAN on the raw images against using pre-processing techniques such as extracting the HOG features. 

\begin{figure*}
\centering
\includegraphics[width=0.9\textwidth]{figures/DBScan_aachen.JPG}
\caption{Test images (left column) from the Aachen data and corresponding segmentation masks (right column) using DBSCAN with eps of 0.3 and minimum points of 100. Each color in the segmentation mask represents a different cluster.}
\label{fig:dbs_aach}
\end{figure*}

## Supervised Methods
Supervised algorithms were performed using the fine annotations of the Cityscapes dataset. Two methods of completing pixel-level segmentation were completed. In this case, multiple objects are within the images. The first approach entails identifying the objects and finding the corresponding pixel for the object via model chaining. The second and more direct approach involves training on the pixel information and obtaining the pixel level map using deep neural network architecture.

### Methods
\textbf{Method 1: Model chaining} involved using Google Colab to input raw, unlabeled (n=10) Cityscapes images, \ref{fig:DETR4}A, into the DEtection TRansformer (DETR) model (pre-trained on the COCO dataset) \cite{carion2020end}. Each of the images were from the Nuremburg dataset. Primary outputs of the model include images with bounding boxes labeled, \ref{fig:DETR4}B, and a list of bounding boxes. In the future, the model outputs will then be inputted into the Segment My Object (SegMyO) pipeline which takes bounding box information and applies masks to the raw image using Mask R-CNN \cite{delearde2021segment}. DETR models were compared for performance and computation time to best select a model for use with SegMyO. Predicted outputs of the entire system are final pixel-level labeled images that can be evaluated against the Cityscapes fine annotated data.

\begin{figure*}
\centering
\includegraphics[width=0.9\textwidth]{figures/Image4Results.pdf}
\caption{Sample comparison between DETR Resnet 50 and original image. Bounding boxes surround the identified objects. Each is labeled appropriately and shows the resulting confidence score.}
\label{fig:DETR4}
\end{figure*}

\textbf{Method 2: Deep Neural Network} utilizes the annotation masks to train the model uses the architectures like RESNET50 or VGG-16 \cite{deeplabv2} to create the masks for the object instances. The latest development of algorithms in CNN and DL pushed to create faster and lighter (computational) networks which create segmentation masks. We tested the pre-trained neural network models of DeepLab on the Cityscapes dataset and compare the pixel-level segmentation performances between various architectures\cite{panoptic_deeplab_2020}. One objective was to identify any performance advantages when using the transformers combined with neural networks. The models 
DeepLab\cite{deeplabv2} was implemented as an end to end pre-trained neural network on Cityscapes to compare different architectures and seeing qualitative measure comparisons.


### Model Architecture Descriptions
\begin{enumerate}
      \begin{enumerate}
        \item \textbf{DETR}: It approaches object detection as a direct set prediction problem. It consists of a set-based global loss, which forces unique predictions via bipartite matching, and a Transformer encoder-decoder architecture. Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel. Due to this parallel nature, DETR is very fast and efficient\cite{detr}.
        \item \textbf{SegMyO}: It automatically extracts the segmented objects in images based on given bounding boxes. When provided with the bounding box, it looks for the output object with the best coverage, based on several geometric criteria. Associated with a  semantic segmentation model trained on a similar dataset(PASCAL-VOC and COCO) this pipeline provides a simple solution to segment efficiently a dataset without requiring specific training, but also to the problem of weakly-supervised segmentation. This is particularly useful to segment public datasets available with weak object annotations coming from an algorithm (in our case DETR).
        \item \textbf{Panoptic-DeepLab}: It is a state-of-the-art box-free system for panoptic segmentation, where the goal is to assign a unique value, encoding both semantic label and instance ID, to every pixel in an image. The class-agnostic instance segmentation is first obtained by grouping the predicted foreground pixels (inferred by semantic segmentation) to their closest predicted instance centers. To generate final panoptic segmentation, we then fuse the class-agnostic instance segmentation with semantic segmentation by the efficient majority-vote scheme.\cite{panoptic_deeplab_2020}
        \item \textbf{Axial-DeepLab}: It incorporates the powerful axial self-attention modules, also known as the encoder of Axial Transformers, for general dense prediction tasks. The backbone of Axial-DeepLab, called Axial-ResNet, is obtained by replacing the residual blocks in any type of ResNets with our proposed axial-attention blocks. They adopt the hybrid CNN-Transformer architecture, where they stack the effective axial-attention blocks on top of the first few stages of ResNets. This hybrid CNN-Transformer architecture is very effective on segmentation tasks\cite{axial_deeplab_2020}.
    \end{enumerate}
\end{enumerate}

\begin{figure*}
\centering
\includegraphics[width=0.9\textwidth]{figures/deepLab_combine_images.png}
\caption{Test images (left column) from the Aachen data, ground truth labels (center column) from the dataset, and corresponding segmentation masks (right column) using DeepLab algorithm. The mask is panoptic which has segmentation mask combined with color variation per instance.}
\label{fig:deeplab_results}
\end{figure*}


### Quantitative Metrics

1.  \textbf{Mean Intersection over Union (mIoU)} is another method to evaluate the predictions from an image segmentation model. This is a metric that takes the IoU over all of the classes and takes the mean of them. This is a good indicator of how well an image segmentation model performs over all the classes that the model would want to detect.
2. \textbf{Computational Time} is an important metric of evaluation. The age of cloud computing, the focus of researchers are moving towards accuracy, but in reality compute is still scarce and is a matter of concern. Hence we will also be looking into determining time efficient methods by using the inbuilt tools to calculate. 
3. \textbf{Object count} is a metric of evaluation for the object recognition algorithm used in Method 1. The simple metric can help us keep on track and measure the performance of the algorithm. 

## Results

Definitions of the metrics used to report results in tables and images include:
\begin{enumerate}
    \item Total Run Time refers to the duration of the code per image iteration.
    \item Model Run Time refers to the duration of running the model per image iteration.
    \item Mean Score refers to the score per image iteration
    \item Mean Score Per Car refers to the score of all the cars per image iteration.
    \item Cars Labeled refers to the number of cars labeled in the image
    \item Items Labeled refers to the number of items labeled in the image.
\end{enumerate}


Fig \ref{fig:deeplab_results} show that DeepLab identifies all the semantic classes and all instances(additionally). Table \ref{table:deepLabmIoU} shows that both Panoptic and Axial algorithms give an mIoU greater than 0.9 for Aachen dataset. Few of the images perform slightly better for Panoptic and others for Axial. We need to do more extensive testing on the total validation dataset to give a greater sense of impact of algorithms on data.
Few features like small objects can be better recognized in few architecture than others. 

Fig \ref{fig:DETR4}, \ref{fig:DETR8}, and \ref{fig:DETR10} help us qualitatively understand the impact of the DETR algorithm. The bounding boxes help recognize the objects in the Cityscapes dataset. This is particularly interesting as DETR was trained on COCO dataset and has not been trained on Cityscapes data before running it on validation. Only items identified by the DETR algorithms with confidence scores of 0.9 or above were kept.

The tables \ref{table:Runtime}, \ref{table:scores}, and \ref{table:itemCount} help us see which RESNET model is better and the comparison of the architecture helps us better to make the network better with latest additions. A relatively constant duration of about 0.7 seconds was found across each DETR model based on the difference between Total Time and Model Time in \ref{table:Runtime}.


\begin{table}[h]
\begin{center}
\caption{Mean Run time per Image Iteration for Each DETR Architecture}
\begin{tabular}{|c||c c ||} 
 \hline
 Model & Total Run Time & Model Run Time\\ [0.5ex] 
 \hline\hline
 Resnet 101 & 9.459 & 8.756 \\ 
 \hline
 Resnet 101-DC5 & 20.445 & 19.744 \\
 \hline
  Resnet 50-DC5 & 18.169 & 17.467 \\
 \hline
 Resnet 50 & 6.359 & 5.655\\
 \hline
\end{tabular}
\label{table:Runtime}

\end{center}
\end{table}

\begin{table}[h]
\begin{center}
\caption{Mean Confidence Scores by Each DETR Architecture}
\begin{tabular}{|c||c c||} 
 \hline
 Model & Mean Score & Mean Score Per Car\\ [0.5ex] 
 \hline\hline
 Resnet 101 &  0.971 & 0.973 \\ 
 \hline
 Resnet 101-DC5 & 0.975 & 0.977 \\
 \hline
  Resnet 50-DC5 & 0.972 & 0.975  \\
 \hline
 Resnet 50  & 0.972 & 0.974 \\
 \hline
\end{tabular}
\label{table:scores}
\end{center}
\end{table}

\begin{table}[h]
\begin{center}
\caption{Number of Objects Labeled by Each DETR Architecture}
\begin{tabular}{|c||c | c||} 
 \hline
 Model & Cars Labeled  & Items Labeled\\ [0.5ex] 
 \hline\hline
 Resnet 101 &  126 & 185 \\ 
 \hline
 Resnet 101-DC5 & 133 & 190 \\
 \hline
  Resnet 50-DC5 &  134 & 202 \\
 \hline
 Resnet 50 & 121 & 184\\
 \hline
\end{tabular}
\label{table:itemCount}
\end{center}
\end{table}

\begin{table}[h]
\begin{center}
\caption{mIoU for DeepLab Models}
\begin{tabular}{|c||c | c||} 
 \hline
Images & Panoptic-DeepLab & Axial-DeepLab\\ [0.5ex] 
 \hline\hline
Image 1 &  0.9364 & 0.9393 \\ 
 \hline
Image 2 & 0.9348 & 0.9307 \\
 \hline
Image 3 & 0.9271 & 0.9208 \\
 \hline
Image 4 & 0.9133 & 0.9240\\
 \hline
\end{tabular}
\label{table:deepLabmIoU}
\end{center}
\end{table}

\begin{figure*}
\centering
\includegraphics[width=0.9\textwidth]{figures/Image10Results.pdf}
\caption{Comparison of the four DETR processed images against the original image from the Neuremberg dataset. Bounding boxes surround the identified objects. Each is labeled appropriately and shows the resulting confidence score.}
\label{fig:DETR10}
\end{figure*}

\begin{figure*}
\centering
\includegraphics[width=0.9\textwidth]{figures/Image8Results.pdf}
\caption{Comparison of the four DETR processed images against the original image from the Neuremberg dataset. Bounding boxes surround the identified objects. Each is labeled appropriately and shows the resulting confidence score.}
\label{fig:DETR8}
\end{figure*}




### Discussion

We observed that the DETR performance for smaller items increases with the DC-5 models as the resolution is increased by a factor of 2, correlating with increases in Items Labeled in Table \ref{table:itemCount}. On the down side, the DC-5 models have a higher computation cost because of the higher cost associated with self-attentions of the encoders, as shown in Table \ref{table:Runtime}. This observation matches with the claim made in the DETR paper\cite{detr}.

Taking the unorthodox way, we are trying to combine two non-related models and without linking them, but rather having the output of one be processed into being input for the other algorithm. We are yet to experiment with the full Model Chaining approach and identify advantages of the system. We are optimistic because the DETR output is bounding boxes of objects in an image, and the input for segmentation for SegMyO are also bounding boxes. We need to get object wise mask, and take a union of the masks to create a bigger image mask. We can evaluate with the ground truth masks once the total image mask is created. 

Future work will involve comparing the DETR models against each other image by image using all the metrics and also further qualitative analysis. This will aid in determining missed or incorrect labels. Due to its high score, relative items labeled count, and low computation time, the Resnet 50 model appears favorable for chaining with SegMyO. With that said, a final decision will be made upon the completion of the further analyses.

DeepLab results (Figure \ref{fig:deeplab_results}) clearly shows that the model is successful qualitative and the (Table \ref{table:deepLabmIoU}) shows the model is successful quantitatively on average. The discussion for the DeepLab is between using different architectures. The RESNET-50 being backbone is a rather famous architecture, but we also extend our implementation to use the axial-transformers combined with ResNets. We are planning to run the model on the total validation dataset and complete the analysis on the DeepLab models trained on Cityscapes. 







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


