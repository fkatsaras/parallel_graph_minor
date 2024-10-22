#!/bin/bash

# Determine directories
TEST_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(dirname "$TEST_DIR")"
SRC_DIR="${ROOT_DIR}/src"

# Function to run a program and measure its execution time
run_and_time() {
    local executable=$1
    local arg1=$2
    local arg2=$3
    local exec_path="${SRC_DIR}/${executable}"

    # Ensure the executable has the necessary permissions and run it
    chmod +x "$exec_path" || {
        echo "Error: Failed to set permissions for $exec_path"
        exit 1
    }

    # Run the executable and capture the output
    local output
    output=$("$exec_path" "$arg1" "$arg2" 2>&1) || {
        echo "Error: Failed to run $exec_path. Output:"
        echo "$output"
        exit 1
    }

    # Log the full output for debugging
    echo "Output from $exec_path:"
    echo "$output"

    # Return the output for time measurement
    echo "$output"
}

# Check if enough arguments are provided
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <implementation> <matrix_A> <matrix_B>"
    echo "Example: $0 openmp ./matrices/e40r2000.mtx ./matrices/e40r2000.mtx"
    exit 1
fi

# Implementation to compare against the serial version
IMPL=$1
MATRIX_A=$2
MATRIX_B=$3

# Run and time matrix multiplication
echo "Running matrix multiplication validation..."

# Run serial implementation and capture time
echo "Running serial implementation..."
serial_output=$( { time run_and_time "serial_mul.exe" "$MATRIX_A" "$MATRIX_B"; } 2>&1 )

# Extract execution time from the time command output
serial_mul_time=$(echo "$serial_output" | grep -oP '(?<=real\s)\d+m\d+\.\d+')
serial_mul_time=$(echo "$serial_mul_time" | awk -F'm' '{ print $1 * 60 + $2 }') # Convert to seconds

# Run the specified parallel implementation and capture time
echo "Running $IMPL implementation..."
impl_output=$( { time run_and_time "${IMPL}_mul.exe" "$MATRIX_A" "$MATRIX_B"; } 2>&1 )

# Extract execution time from the time command output
impl_mul_time=$(echo "$impl_output" | grep -oP '(?<=real\s)\d+m\d+\.\d+')
impl_mul_time=$(echo "$impl_mul_time" | awk -F'm' '{ print $1 * 60 + $2 }') # Convert to seconds

# Compare execution times
echo "Comparing execution times..."
echo

# Print the execution times
echo "$IMPL: $impl_mul_time seconds"
echo "Serial: $serial_mul_time seconds"

# Compare times using Bash's built-in arithmetic
if (( $(echo "$impl_mul_time < $serial_mul_time" | bc -l) )); then
    diff=$(echo "$serial_mul_time - $impl_mul_time" | bc -l)
    echo "$IMPL implementation is faster than the serial implementation by $diff seconds"
elif (( $(echo "$impl_mul_time > $serial_mul_time" | bc -l) )); then
    diff=$(echo "$impl_mul_time - $serial_mul_time" | bc -l)
    echo "Serial implementation is faster than the $IMPL implementation by $diff seconds"
else
    echo "Execution times are the same."
fi
