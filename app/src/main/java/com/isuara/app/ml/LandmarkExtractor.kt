package com.isuara.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker

/**
 * Extracts normalized landmarks from a single camera frame.
 * Uses MediaPipe PoseLandmarker.
 */
class LandmarkExtractor(context: Context) {

    companion object {
        private const val TAG = "LandmarkExtractor"
        private const val MODEL_PATH = "pose_landmarker_full.task"
    }

    private var poseLandmarker: PoseLandmarker? = null

    init {
        val baseOptions = BaseOptions.builder()
            .setModelAssetPath(MODEL_PATH)
            .setDelegate(Delegate.GPU)
            .build()

        val options = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(baseOptions)
            .setRunningMode(RunningMode.VIDEO) // Process video stream
            .setNumPoses(1)
            .build()

        poseLandmarker = PoseLandmarker.createFromOptions(context, options)
        Log.i(TAG, "PoseLandmarker created")
    }

    /**
     * @return 56 keypoints (x,y) for body + hands, or null on failure
     */
    fun extract(bitmap: Bitmap, timestampMs: Long): FloatArray? {
        try {
            val image = BitmapImageBuilder(bitmap).build()
            val result = poseLandmarker?.detectForVideo(image, timestampMs) // Synchronous call

            if (result != null && result.landmarks().isNotEmpty()) {
                val allLandmarks = mutableListOf<Pair<Float, Float>>()

                // Pose landmarks (33)
                result.landmarks()[0].forEach {
                    allLandmarks.add(it.x() to it.y())
                }
                // TODO: Add hand landmarks if available

                // Flatten to FloatArray
                return allLandmarks.flatMap { listOf(it.first, it.second) }.toFloatArray()
            }
        } catch (e: Exception) {
            Log.e(TAG, "Pose landmark detection failed", e)
        }
        return null
    }

    fun close() {
        poseLandmarker?.close()
        Log.i(TAG, "PoseLandmarker closed")
    }
}
