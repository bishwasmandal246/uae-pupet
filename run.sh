#!/bin/bash

for i in {1..25}
do
   python PUPET-Official/main.py -d MNIST -g UAE -p 40 -o false
   echo "Iteration: $i complete!"
done

for i in {1..25}
do
   python PUPET-Official/data-type-aware-main.py -g UAE
   echo "Iteration: $i complete!"
done
