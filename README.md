#Emote
This is source for a FACS recognition system.

##Introduction
Emote is a toolkit for image processing, facial detection, and facial action code recognition. All the code written to preprocess data sets is included in reuseable modules.

##Setup
###Dependencies
The precompiled dependencies for nikola.dartmouth.edu are available on discovery at /idata/lchang/Resources/emote-deps. The README in that folder gives some more instructions on how to use the dependencies.

- OpenCV 3.0+ (w/ python and contrib)
- DLib (w/ python)
- numpy 1.7+
- scikit
- matplotlib
- tensorflow


###config.json
Updating the configuration file for your system is necessary due to varying locations of data across machines. We also don't like keeping large datasets in the repo, so this allows us to point Emote to the proper locations.

Currently config.json has three main sections, `data`, `model`, and `detector`.

####`data`
This section is indexed by data set (as their names are listed in `util.constants.py`. It includes the location of the dataset on the current disk as well as the image size of the dataset. This is done because locations of data sets may differ across machines.

Here's an example:

    "ck":{
          "location":"data/ck_cleaned",
          "image_size":96
    },
    "difsa":{
      "location":"data/DIFSA/",
      "image_size":96
    },
    
####`model`
The model section tells Emote which expression model to use and how to configure it. 
`facs` - Includes a list of AUs to search/train on, and a boolean to determine if intensities should be considered. 

`data` - Name of a data set (as found in util.constants) to train the model on, or which the saved model was trained on.

`name` - The name of the model you want to use

`image_size`- (int) rectangular image dimension to be used with the model.

####`detector`
The detector class you want to use to find faces. There's only one right now and it does a pretty good job, `haar`

###Expected Contents of Data Set Folders:
####CK+
Cohn-Kanade + data set. There is on file with all the AUs called `fac_data`. Each line in the file represents the AUs for on image. The image name is the first column, then pairs of AUs and intensity follow as so:

	<SubjectNum> <AU> <Intensity> <AU> <Intensity> <AU> <Intensity> ...

Each line corresponds to a single image and each `<AU> <Intensity>` pair correspond to one another.

Found in: `/idata/lchang/Data/FACS_data/ck_clean` on discovery. The corresponding images should be in the same directory.

####DIFSA
Cleaned and expanded DIFSA dataset. Two directorys are included: `AUs` and `Images`. `AUs` includes 27x4 CSV files containing data for each processed video. The CSV files contain a row for each video frame. The format is:

    Frame | 1 | 2 | 4 | 5 | 6 | 9 | 12 | 15 | 17 | 20 | 25 | 26

The values of the columns other than the first represent the intensity of the AU if present. The scale is 0-5, where 0 is not present, and 5 is the highest intensity.

The `Images` folder contains directories which mirror the names of the CSV files in the `AUs` directory. Each directory contains the individual frames of the source videos. The frames are have the same name of the parent directory, though they are appended with the frame number. For example, for the `LeftSN004 video`, the CSV will be named `LeftSN004`, the directory containing its frames will be `LeftSN004`, and the 45th frame will be inside named `LeftSN004_45`.

The DIFSA set can be found at `/idata/lchang/Data/FACS_data/DIFSA_clean`. It includes both left and right camera data as well as their mirror equivalents. `DIFSA_nonzero` is scrubbed of any frames which have no AUs and is a little more space conscious

##How to Use
Run `python emote.py --help` for more information

##To Do

1. No model is working particularly well.... Definitely more research to be done there

2. I increased the size of DIFSA by mirroring, and it doesn't fit on AFS anymore b/c its 19GB so I can't run it on Nikola. More disk space very necessary.

3. data.repositories.DIFSARepository loads entire dataset into memory. Should be converted to only load specific files at once


