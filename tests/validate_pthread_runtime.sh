#!/bin/bash

TEST_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(dirname "$TEST_DIR")"
SRC_DIR="${ROOT_DIR}/src"

# Function to run a program and measure its execution time
# Arguments:
# 	$1 - Executable name
# 	$2 - Argument (matrix) 1 for the executable
# 	$3 - Argument (matrix) 2 for the executable
run_and_time() {
	local executable=$1
	local arg1=$2
	local arg2=$3

	# echo "Running $executable with arguments $arg1 and $arg2..."

	# Run the executable and capture the output
	local output
	output=$("${SRC_DIR}/${executable}" $arg1 $arg2)

	# Extract the execution time 
	local exec_time
	exec_time=$(echo "$output" | grep "Total multiplication execution time: " | grep -oP '\d+\.\d+')

	# echo "$executable execution time: ${exec_time}ms"

	exec_float=$(echo "$exec_time" | bc)

	# Return elapsed time converted to float
	echo $exec_float
}

# Matrix arguments
MATRIX_A="${TEST_DIR}/A.mtx.sty"
MATRIX_B="${TEST_DIR}/B.mtx.sty"
MATRIX_C="${TEST_DIR}/C.mtx.sty"
NUM_CLUSTERS=2

# Run and time matrix multiplication
echo "Running matrix multiplication validation..."

# Run serial implementation
echo "Running serial implementation..."
serial_mul_time=$(run_and_time "serial_mul" "$MATRIX_A" "$MATRIX_B" "true")

# Run pthreads implementation
echo "Running pthread implementation..."
pthread_mul_time=$(run_and_time "pthread_mul" "$MATRIX_A" "$MATRIX_B" "true")

# Compare execution times
echo "Comparing execution times..."
echo

# Use bc for floating point comparison
if [ "$(echo "$pthread_mul_time < $serial_mul_time" | bc -l)" -eq 1 ]; then
    diff=$(echo "$serial_mul_time - $pthread_mul_time" | bc -l)
    echo "pThreads implementation is faster than the serial implementation by $diff seconds"
    echo "pThreads: $pthread_mul_time seconds"
    echo "Serial: $serial_mul_time seconds"
	exit 0
elif [ "$(echo "$pthread_mul_time > $serial_mul_time" | bc -l)" -eq 1 ]; then
    diff=$(echo "$pthread_mul_time - $serial_mul_time" | bc -l)
    echo "Serial implementation is faster than the pThreads implementation by $diff seconds"
    echo "pThreads: $pthread_mul_time seconds"
    echo "Serial: $serial_mul_time seconds"
	exit 1
else
    echo "Execution times are the same."
    echo "pThreads: $pthread_mul_time seconds"
    echo "Serial: $serial_mul_time seconds"
	exit 1
fi
