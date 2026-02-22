package com.isuara.app.ml

import android.content.Context
import android.util.Log
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.GpuDelegate
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * SignInterpreter — TFLite wrapper for the BIM sign language model.
 *
 * Input:  (1, 30, 780) float32
 * Output: (1, 98)      float32 softmax probabilities
 *
 * Tries NnApiDelegate (NPU) first, falls back to GpuDelegate, then CPU.
 */
class SignInterpreter(context: Context) {

    companion object {
        private const val TAG = "SignInterpreter"
        private const val MODEL_FILE = "bim_lstm_v3_int8.tflite"
        private const val NUM_CLASSES = 98
        private const val SEQUENCE_LENGTH = 30
        private const val NUM_FEATURES = 780
    }

    private val interpreter: Interpreter
    private var gpuDelegate: GpuDelegate? = null
    private val outputBuffer: Array<FloatArray> = Array(1) { FloatArray(NUM_CLASSES) }

    init {
        val model = loadModelFile(context)
        var tempInterpreter: Interpreter? = null

        // Attempt 1: Try with GPU Delegate
        try {
            // Check if the device actually supports the GPU delegate
            val compatList = org.tensorflow.lite.gpu.CompatibilityList()

            if (compatList.isDelegateSupportedOnThisDevice) {
                // Gets the optimal, non-deprecated Options for this specific phone
                val delegateOptions = compatList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)

                val gpuOptions = Interpreter.Options().apply { numThreads = 4 }
                gpuOptions.addDelegate(gpuDelegate!!)

                tempInterpreter = Interpreter(model, gpuOptions)
                Log.i(TAG, "Using GPU delegate")
            } else {
                Log.w(TAG, "GPU not supported on this device. Skipping to CPU fallback.")
            }
        } catch (e: Throwable) {
            Log.w(TAG, "GPU delegate failed or rejected the model. Falling back to CPU: ${e.message}")
            gpuDelegate?.close()
            gpuDelegate = null
        }

        // Attempt 2: Fallback to CPU
        if (tempInterpreter == null) {
            val cpuOptions = Interpreter.Options().apply { numThreads = 4 }
            tempInterpreter = Interpreter(model, cpuOptions)
            Log.i(TAG, "Using CPU delegate fallback")
        }

        interpreter = tempInterpreter
        Log.i(TAG, "Model loaded: $MODEL_FILE")

        // Validate shapes
        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()
        Log.i(TAG, "Input shape: ${inputShape.contentToString()}")
        Log.i(TAG, "Output shape: ${outputShape.contentToString()}")
    }

    /**
     * Run inference on a preprocessed sequence.
     *
     * @param features FloatArray of size 30 × 780 = 23400, row-major
     * @return FloatArray of 98 softmax probabilities
     */
    fun predict(features: FloatArray): FloatArray {
        require(features.size == SEQUENCE_LENGTH * NUM_FEATURES) {
            "Expected ${SEQUENCE_LENGTH * NUM_FEATURES} features, got ${features.size}"
        }

        // Prepare input buffer [1, 30, 780]
        val inputBuffer = ByteBuffer.allocateDirect(4 * SEQUENCE_LENGTH * NUM_FEATURES).apply {
            order(ByteOrder.nativeOrder())
            rewind()
            asFloatBuffer().put(features)
        }

        interpreter.run(inputBuffer, outputBuffer)
        return outputBuffer[0]
    }

    /**
     * Get predicted class index and confidence.
     */
    fun predictTopClass(features: FloatArray): Pair<Int, Float> {
        val probs = predict(features)
        var maxIdx = 0
        var maxVal = probs[0]
        for (i in 1 until probs.size) {
            if (probs[i] > maxVal) {
                maxVal = probs[i]
                maxIdx = i
            }
        }
        return Pair(maxIdx, maxVal)
    }

    fun close() {
        interpreter.close()
        gpuDelegate?.close()
    }

    private fun loadModelFile(context: Context): MappedByteBuffer {
        val fd = context.assets.openFd(MODEL_FILE)
        val inputStream = FileInputStream(fd.fileDescriptor)
        val channel = inputStream.channel
        return channel.map(FileChannel.MapMode.READ_ONLY, fd.startOffset, fd.declaredLength)
    }
}
