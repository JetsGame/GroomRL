#!/bin/bash

IP=`hostname --ip-address`
PORT=1234
echo $IP:$PORT
mongod --dbpath ./db --port $PORT --noprealloc
