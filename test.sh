#!/bin/bash
set -e

INPUT_DIR="$(pwd)/input"
MODEL_CACHE="$HF_CACHE"
DOCKER_IMAGE="nsfw-image-detection"

# Ensure input images exist
if [[ ! -f "$INPUT_DIR/sfw.jpg" ]] || [[ ! -f "$INPUT_DIR/nsfw.jpg" ]]; then
    echo "Error: Missing test images in $INPUT_DIR. Place sfw.jpg and nsfw.jpg there."
    exit 1
fi

run_test() {
    local device=$1
    local image=$2
    local expected_label=$3

    echo "Running inference on $image using $device..."
    START_TIME=$(date +%s.%N)

    if [[ "$device" == "cpu" ]]; then
        GPU_FLAG=""
    else
        GPU_FLAG="--gpus all"
    fi

    OUTPUT=$(docker run \
        -v "$MODEL_CACHE:/app/models" \
        -e HF_TOKEN="$HF_TOKEN" \
        --rm \
        $GPU_FLAG \
        -v "$INPUT_DIR:/app/input" \
        $DOCKER_IMAGE --image "/app/input/$image" | tee /dev/tty)

    END_TIME=$(date +%s.%N)
    EXECUTION_TIME=$(echo "$END_TIME - $START_TIME" | bc)

    echo "$device Execution Time: ${EXECUTION_TIME} seconds"

    # Validate output
    if echo "$OUTPUT" | grep -qi "$expected_label"; then
        echo "Test passed: $image classified correctly as $expected_label"
    else
        echo "Test failed: $image classification did not match expected ($expected_label)"
        exit 1
    fi
}

# Run tests
run_test "cpu" "sfw.jpg" "normal"
run_test "gpu" "sfw.jpg" "normal"
run_test "cpu" "nsfw.jpg" "nsfw"
run_test "gpu" "nsfw.jpg" "nsfw"
