##Introduction

At the end of my senior year of college I began working with some neuroscientists at Dartmouth in order to gain some academic research experience and to have a good reason to learn about some new technology. The [Computational Social Affective Neuroscience Laboratory](https://cosanlab.com) (COSAN) is run by Professor Luke Chang. They study emotions and their role in social situations from a computational neuroscience angle.

I was interested in that work that COSAN was doing, but didn't know much about psychology or neuroscience, so I offered my engineering skills if there was something to be done. COSAN was about to begin running experiments on involving filming subjects faces while they watched episodes of *Friday Night Lights* the hit NBC television drama. One thing that they thought would be nice to have is a pipeline that can analyze video of subjects for facial action codes, which could be analyzed further and associated with specific scenes or emotions happening on screen.

This project was meant to be that pipeline though it is very much overdue.

##Facial Action Codes

[The facial action coding system](https://en.wikipedia.org/wiki/Facial_Action_Coding_System) is a standardized way of measuring human facial movements. The latest system specifies 46 codes in total; each corresponds to the movement of a muscle or group of muscles in the face. Paul Ekman was involved in creating the modern system and you can buy a manual on how to read FACs from [his website](http://www.paulekman.com/product-category/facs/).

An *action unit* is meant to represent a single muscle or action of the face. Each unit is numbered and corresponding description. For example, AU4 is the brow lowerer action and involves four facial muscles (depressor glabellae, depressor supercilii, corrugator supercilii) ([Wikipedia](https://en.wikipedia.org/wiki/Facial_Action_Coding_System)). AUs are rated on a scale from 0-5. 0 being not present, and 5 being the most prominent. Most researchers working with AU datasets require an AU rating the be at least 2 in order to consider it present on the face. An action unit does not correspond to a particular emotion. Groups of action units which, when expressed together, can indicate a presence of a basic emotion being conveyed by the face. For example, sadness tends to be indicated by the presence of AUs 1, 4, and 15, or fear requires AUs 1, 2, 4, 5, 7, 20, and 26 ([Wikipedia](https://en.wikipedia.org/wiki/Facial_Action_Coding_System)).

There are many databases composed of action unit labeled images and videos which are available to researchers, but many of them tend to lack a variety of situations and subjects. The first roadblock is getting subjects to display a variety of action units in a natural setting. When facial expressions are posed (not natural), they tend to be different from non-posed expressions []. Natural facial expressions are likely to be more subtle than posed ones, and the difference could affect the reliability of some models. Some databases have subjects watch videos or commercials in order to elicit a natural response that can be labeled afterwards [[DIFSA](http://mohammadmahoor.com/databases/denver-intensity-of-spontaneous-facial-action/)] [[AM-FED](http://www.affectiva.com/facial-expression-dataset/)]. These datasets are more difficult to create, and the frequency of AUs are very difficult to control. Subjects also tend to be in very controlled environments; labeled AU data taken from uncontrolled environments is rare.

##Related work

Action units were adopted by Paul Ekman and Wallace Friesen in 1978 [Tech for measurement], but a major update to the system was published in a 2002 by Ekman and Hager [FAC system]. Much of research on automatic facial action code recognition has utilized either the 1978 version or 2002 version.

An image's signal to noise ratio is extremely small, so the first step, as with most computer vision problems, is to preprocess any data you are working on and attempt to select features. Converting images to grayscale and balancing brightness is ubiquitous in AU preprocessing pipelines. Many works detect faces, crop, and resize them to completely isolate the face being operated on. Often a small transformation is applied to a face in order to properly align its features to common areas in the image like leveling the eyes, and vertically aligning the nose. The subtly of AUs seem to necessitate heavy preprocessing in order to maximize a result. I have not seen any instances of a model being directly applied to natural images with good results.

In terms of feature selection and model approaches, much of the recent work can be divided into three groups: support vector machines, boosting algorithms, and neural networks. Bartlett used sets of Gabor filters to extract relevant features from training images, and utilized both AdaBoot and SVMs to classify over 27 AUs. They used this approach to perform fully automatic AU recognition at 24 frames/second. In more recent years, researchers have applied deep neural networks to the problem of facial expression recognition. Gudi used 3 convolutional layers and a fully connected layer trained on a negative log-likelihood loss. They interpretted the outputs of the network as AU confidence levels, then determined an optimal decision threshold based on the F1 score of their validation dataset. Ghosh used 2 convolutional layers with 2 fully connected layers trained on a multi-label softmax classification loss function. They also tacked a quadratic classifier on the end of the network to produce the final classifications.

##Methods
For this project we were not aiming to break any new ground. We only wanted a pipeline that worked and could be used in the lab and was good enough to perform actual analysis on the data that the model would spit out. With this goal in mind, we began looking for data to use, and a paper that would be feasible to emulate in-house.

###Data
There are many datasets available for research use right now, but many are difficult to get access to. These datasets aren't created every year, and the websites that are meant to distribute them are not kept up-to-date. Many websites were last updated in the early 2000s, and were accompanied by dead links and unattended email addresses. The ones I talk about below were the most promising candidates.

The most popular dataset, [Cohn-Kanade+](http://www.consortium.ri.cmu.edu/ckagree/) (CK+), was easily accessed and downloaded. The data are series of images of a subject expressing a particular AU. For an AU there are 5 or so images taken one after another and the AU is meant to get more pronounced over time. CK+ is a standard dataset for AU recognition. Unfortunately it is a mixture of posed and non-posed expressions making the data unrealistic. It is also very sparse. The number of AU examples are in the hundreds; a number too low for a deep neural network. CK+ would be a good testing set, but we would need something more substantial for training.

A more recent and comprehensive database is the [Affectiva-MIT Facial Expression Dataset](http://www.affectiva.com/facial-expression-dataset/) (AM-FED). It consists of 242 videos taken by users of a website as they watched three Super Bowl ads. All facial expressions are spontaneous, and there are a wide variety of environments. AU labels are assigned to each frame as well as head movements, smile, and general expressiveness information.

The [Denver Intensity of Spontaneous Facial Action Database](http://mohammadmahoor.com/databases/denver-intensity-of-spontaneous-facial-action/) (DIFSA) contains 27 videos of subjects watching a four minute montage of nine video clips from YouTube that were selected based on their anticipated emotional response. Each subject is recorded in a controlled environment for roughly the same period of time. Each frame is manually labeled for 12 AUs and 66 facial landmarks.

We decided to work primarily with the DIFSA dataset as its raw form was very easy to begin processing right away. AM-FED also seemed like a great choice, but the raw data was labeled not by frame, but by which frame the AU changed, making it a tedious job to separate out each frame and match it with the proper AUs.

###Preprocessing
Our preprocessing was relatively standard and pulled from techniques that we had seen in other papers. We used [OpenCV](http://opencv.org/) on each frame to get the bounding box for the face. The face was then grayscaled and resized to 96x96 pixels. Afterwards we used [OpenFace](https://cmusatyalab.github.io/openface/) to perform an affine transformation based on facial landmarks to standardize the alignment of faces in the cropped images.

The DIFSA dataset is extremely skewed towards negative AU examples - no AUs are present in a frame. In order to account for this, I performed a vertical mirroring of each frame to create new data then filtered out 88% of the AU absent frames from the set. This left us with about 50% of data being void of AUs and the other 50% having at least one instance. This helps to prevent a model bias towards only predicting zeros. In total, our cleaned dataset comprised of 60,000+ frames. Divided 70-15-15 for training, validation, and testing sets respectively.

###Data format
DIFSA is divided up into `.cvs` files for each subject with sets of AU labels for each frame. These were accompanied by videos of each subject from two camera perspectives. To perform the preprocessing, we saved individual frames to `.png` organized by subject and camera. This structure made it complicated and time consuming to match data in `.csv` files with their corresponding frame. To make the data easier to work with, we moved each example/label pair into Google's [Protocol Buffer](https://developers.google.com/protocol-buffers/) format.

The protobuf definition that we used looked like this:
```
package emote_data;

message Face {
    optional string id = 1;
    required bytes image = 2;
    repeated int64 aus = 3;
}
```

An ID string was a combination of the subject number, frame, and some other metadata. The frame was easy to represent as a `byte[]`, and the AUs were represented as an array of `int`. Protocol buffers made it significantly easier to load directly into the network, sped up the training process by decreasing batch read times, and saved substantial storage space - the protobuf set is about 1/3 the size.

##Network Architecture

For this project, it was never our goal to do something revolutionary. We needed something that would work and be useful. It is much simpler to follow the beaten path with an architecture that we know can provide good results. Ghosh's work on multi-label AU recognition with convolutional networks fit our needs. In particular, they used a specific multi-label loss function, which allowed us to greatly simplify our code and system.

The network has two convolutional layers, and two fully-connected layers. The convolutional layers are aggregates of a convolutional layer (1x1 stride), a leaky ReLU activation (epsilon of 0.0001), a dropout layer (50% while training), and a max-pooling layer (2x2 pool size). The two fully-connected layers are 500 and 100 nodes in order which connects to a 12 node output layer - DIFSA data have labels for 12 AUs.

###Loss function

Ghosh's loss function is a multi-label binary softmax. The function applies a standard softmax to the output of the network *f(x<sub>i</sub>)* for each AU *f<sub>j</sub>(x<sub>i</sub>)* to get *p<sub>i</sub>*. It multiplies *log(p<sub>i</sub>*) with the label for the AU - 0 or 1. The products for each example are summed, and the sum is averaged and negated over the entire batch.

###Quadratic Discriminant Analysis
To perform the final classification, we train a one-vs-all QDA classifier on the output layer of the network. Each AU is predicted based on the output for all other AUs. The classifier produces a binary classification for each AU, labeling each index as a 0 or 1.

QDA is a probabilistic tool and based on the output *h<sub>i</sub>* of the network and *y<sub>ij</sub>*, the j-th AU in the image's label, learns the posterior probability *P(y<sub>ij</sub> | h<sub>i</sub>)*.

##Implementation
Keras was used to implement the network with a Theano backend. We used scikit-learn's QDA classifier implementation as well as its OneVsRestClassifier to perform the multi-label classification. I will be writing a future post that outlines the code and technical work done for this project. It will be linked *here* when it is posted. 

##Results

##References

P. Ekman and W. Friesen. Facial Action Coding System: A Technique for the Measurement of Facial Movement. Consulting Psychologists Press, Palo Alto, 1978.

Paul Ekman, Wallace V. Friesen, and Joseph C. Hager. Facial Action Coding System: The Manual on CD ROM. A Human Face, Salt Lake City, 2002.

Bartlett, Marian Stewart, et al. "Fully automatic facial action recognition in spontaneous behavior." Automatic Face and Gesture Recognition, 2006. FGR 2006. 7th International Conference on. IEEE, 2006.

Gudi, Amogh, et al. "Deep learning based FACS action unit occurrence and intensity estimation." Automatic Face and Gesture Recognition (FG), 2015 11th IEEE International Conference and Workshops on. Vol. 6. IEEE, 2015.

Littlewort, Gwen, et al. "Dynamics of facial expression extracted automatically from video." Image and Vision Computing 24.6 (2006): 615-625.

Ghosh, Sayan, et al. "A multi-label convolutional neural network approach to cross-domain action unit detection." Affective Computing and Intelligent Interaction (ACII), 2015 International Conference on. IEEE, 2015.
