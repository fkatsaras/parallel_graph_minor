#!/bin/bash

# Determine directories
TEST_DIR="$(dirname "$(realpath "$0")")"
ROOT_DIR="$(dirname "$TEST_DIR")"
SRC_DIR="${ROOT_DIR}/src"

# Function to run a program and measure its execution time
# Arguments:
#   $1 - Executable name
#   $2 - Argument (matrix) 1 for the executable
#   $3 - Argument (matrix) 2 for the executable
run_and_time() {
    local executable=$1
    local arg1=$2
    local arg2=$3
    local exec_path="${SRC_DIR}/${executable}"

    # Check if the executable has the necessary permissions
    if [ ! -x "$exec_path" ]; then
        echo "Error: Permission denied or $exec_path is not executable."
        echo "Attempting to set execute permissions..."
        chmod +x $exec_path

        # Recheck if setting permissions was successful
        if [ ! -x "$exec_path" ]; then
            echo "Error: Failed to set permissions for $exec_path"
            exit 1
        fi
    fi

    # Run the executable and capture the output
    local output
    output=$("$exec_path" $arg1 $arg2 2>&1)

    # Check if the command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Failed to run $exec_path. Output:"
        echo "$output"
        exit 1
    fi

    # Extract the execution time 
    local exec_time
    exec_time=$(echo "$output" | grep "Total multiplication execution time: " | grep -oP '\d+\.\d+')

    # Check if execution time was captured
    if [ -z "$exec_time" ]; then
        echo "Error: Failed to capture execution time from $exec_path."
        exit 1
    fi

    # Convert execution time to float
    exec_float=$(echo "$exec_time" | bc)

    # Return elapsed time converted to float
    echo $exec_float
}

# Matrix arguments
MATRIX_A="../matrices/e40r2000.mtx ../matrices/e40r2000.mtx "
MATRIX_B=../matrices/e40r2000.mtx ../matrices/e40r2000.mtx 
MATRIX_C="${TEST_DIR}/C.mtx.sty"

# Implementation to compare against the serial version
IMPL=$1

if [[ -z "$IMPL" ]]; then
    echo "Please specify the implementation to compare (e.g., openmp or pthread)."
    exit 1
fi

# Run and time matrix multiplication
echo "Running matrix multiplication validation..."

# Run serial implementation
echo "Running serial implementation..."
serial_mul_time=$(run_and_time "serial_mul" "$MATRIX_A" "$MATRIX_B")

# Run the specified parallel implementation
echo "Running $IMPL implementation..."
impl_mul_time=$(run_and_time "${IMPL}_mul" "$MATRIX_A" "$MATRIX_B")

# Compare execution times
echo "Comparing execution times..."
echo

# Use bc for floating point comparison
if [ "$(echo "$impl_mul_time < $serial_mul_time" | bc -l)" -eq 1 ]; then
    diff=$(echo "$serial_mul_time - $impl_mul_time" | bc -l)
    echo "$IMPL implementation is faster than the serial implementation by $diff seconds"
    echo "$IMPL: $impl_mul_time seconds"
    echo "Serial: $serial_mul_time seconds"
    exit 0
elif [ "$(echo "$impl_mul_time > $serial_mul_time" | bc -l)" -eq 1 ]; then
    diff=$(echo "$impl_mul_time - $serial_mul_time" | bc -l)
    echo "Serial implementation is faster than the $IMPL implementation by $diff seconds"
    echo "$IMPL: $impl_mul_time seconds"
    echo "Serial: $serial_mul_time seconds"
    exit 1
else
    echo "Execution times are the same."
    echo "$IMPL: $impl_mul_time seconds"
    echo "Serial: $serial_mul_time seconds"
    exit 1
fi

