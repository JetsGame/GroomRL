Jet grooming with Machine Learning
===============================================

This repository investigates some applications of Machine Learning to
jet grooming.

## Installation

If you are a developer:
```
python setup.py develop # it will link the files from this repo
```
else
```
python setup.py install
```

## Running the code

In order to launch the code run:
```
groomer <runcard.json> --output <folder> [--plot] [--cpp] [--nev n]
```

This will create a folder containg the result of the fit.

To create diagnostic plots, run with the --plot option or use
```
groomer-plot <folder> [--nev n]
```
on the previous output folder.

The groomer can also be exported with the --cpp option, or by running
```
groomer-cpp <folder> [-v]
```

To apply an existing grooming model to a new data set, you can use
```
groomer-apply <folder> <datafile> [--nev n]
```
which will create a new directory in <folder> using the datafile name.

