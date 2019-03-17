GroomRL: jet grooming through reinforcement learning
====================================================

This repository contains the code and results for the .

## Installation

```
python setup.py install
```

## Running the code

In order to launch the code run:
```
groomrl <runcard.json> --output <folder> [--plot] [--cpp] [--nev n]
```

This will create a folder containg the result of the fit.

To create diagnostic plots, run with the --plot option or use
```
groomrl-plot <folder> [--nev n]
```
on the previous output folder.

The groomer can also be exported with the --cpp option, or by running
```
groomrl-cpp <folder> [-v]
```

To apply an existing grooming model to a new data set, you can use
```
groomrl-apply <folder> <datafile> [--nev n]
```
which will create a new directory in `<folder>` using the datafile name.

## References

* paper ref.
