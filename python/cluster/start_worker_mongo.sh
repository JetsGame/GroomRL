#!/bin/bash

IP=hostname --ip-address
PORT=1234
workdir=`pwd`
echo $IP:$PORT
hyperopt-mongo-worker --mongo=$IP:$PORT/groomer --workdir=$workdir
