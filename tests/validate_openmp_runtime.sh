#!/bin/bash

# Function to run a program and measure its execution time
# Arguments:
#       $1 - Executable name
#       $2 - Argument (matrix) 1 for the executable
#       $3 - Argument (matrix) 2 for the executable
#       $4 - Optional argumant for suppressing output (e.g., "true" or "false")
run_and_time() {
        local executable=$1
        local arg1=$2
        local arg2=$3
        local suppress=${4:-"true"}

        echo "Running $executable with arguments $arg1 and $arg2..."

        # Start timing
        local start_time=$(date +%s%N)

        # Run the executable and optionally suppress the output
        if [ "$suppress" = "true" ]; then
                ./$executable $arg1 $arg2 > /dev/null
        else
                ./$executable $arg1 $arg2
        fi

        # End timing
        local end_time=$(date +%s%N)

        # Calculate elapsed time in ms
        local elapsed_time=$(( ($end_time - $start_time) / 1000000 ))

        echo "$executable execution time: ${elapsed_time}ms"

        # Return elapsed time
        echo $elapsed_time

}

# Matrix arguments
MATRIX_A="A.mtx.sty"
MATRIX_B="B.mtx.sty"
MATRIX_C="C.mtx.sty"
NUM_CLUSTERS=2

# Run and time matrix multiplication
echo "Running matrix multiplication validation..."

# Run serial implementation
echo "Running serial implementation..."
serial_mul_time=$(run_and_time "../src/serial_mul" "$MATRIX_A" "$MATRIX_B" "true")

# Run pthreads implementation
echo "Running openmp implementation..."
openmp_mul_time=$(run_and_time "../src/openmp_mul" "$MATRIX_A" "$MATRIX_B" "true")

# Compare execution times
echo "Comparing execution times..."
if [ "$openmp_mul_time" -lt "$serial_mul_time" ]; then
        diff=$((serial_mul_time - pthread_mul_time))
        echo "OpenMP implementation is faster than the serial implementation by ${diff}ms"
else
        diff=$((openmp_mul_time - serial_mul_time))
        echo "Serial implementation is faster than the OpenMP implementation by ${diff}ms"
fi

# Additional validations or checks can be added here 
