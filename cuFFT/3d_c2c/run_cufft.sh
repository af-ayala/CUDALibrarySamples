#!/bin/bash

for i in $(seq 10); do
    nodes=$((2**$i))
    ./3d_mgpu_c2c_example_single $nodes
done
for i in $(seq 10); do
    nodes=$((2**$i))
    ./3d_mgpu_c2c_example_double $nodes
done

for i in $(seq 10); do
    nodes=$((2**$i))
    ./3d_c2c_example_single $nodes
done
for i in $(seq 10); do
    nodes=$((2**$i))
    ./3d_c2c_example_double $nodes
done

for i in $(seq 30); do
    nodes=$((2**$i))
    ./1d_c2c_example $nodes
done

