#Emote
This is source for a FACS recognition system.

##Introduction
Emote is a toolkit for image processing, facial detection, and facial action code recognition. All the code written to preprocess data sets is included in reuseable modules.

##Setup
###Dependencies
Emote has only a few dependencies which are genearlly easy to install:
- OpenCV 3.0+ (w/ python)
- DLib (w/ python)
- numpy 1.7+
- scikit
- matplotlib
- tensorflow

###config.json
Updating the configuration file for your system is necessary due to varying locations of data across machines. We also don't like keeping large datasets in the repo, so this allows us to point Emote to the proper locations.

Currently config.json has only one section - `data` - which is just a mapping between the dataset name, and its directory on the current machine. Some are always required depending on which module they relate to. For some of the larger sets, if the repository already contains the trained model, then you don't need it unless you plan on retraining that model.

###Expected Contents of Data Set Folders:
####CK+
I've been calling this `ck` in the code. It requires a file file which looks like:
	<SubjectNum> <AU> <Intensity> <AU> <Intensity> <AU> <Intensity> ...
Each line corresponds to a single image and each `<AU> <Intensity>` pair correspond to one another

##How to Use
Run `python emote.py --help` for more information
