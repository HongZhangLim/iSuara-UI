package com.isuara.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * LandmarkExtractor — wraps MediaPipe PoseLandmarker + HandLandmarker
 * to extract 258 keypoints per frame.
 *
 * Pose: 33 landmarks × 4 (x, y, z, visibility) = 132
 * Left Hand: 21 landmarks × 3 (x, y, z) = 63
 * Right Hand: 21 landmarks × 3 (x, y, z) = 63
 * Total: 258
 *
 * Uses VIDEO mode for synchronous processing on the ML thread.
 */
class LandmarkExtractor(context: Context) {

    companion object {
        private const val TAG = "LandmarkExtractor"
        private const val POSE_MODEL = "pose_landmarker_lite.task"
        private const val HAND_MODEL = "hand_landmarker.task"
    }

    private val poseLandmarker: PoseLandmarker
    private val handLandmarker: HandLandmarker

    init {
        // ── Pose Landmarker ──
        val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(POSE_MODEL)
                    .setDelegate(Delegate.GPU)
                    .build()
            )
            .setRunningMode(RunningMode.VIDEO)
            .setNumPoses(1)
            .setMinPoseDetectionConfidence(0.5f)
            .setMinPosePresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        poseLandmarker = try {
            PoseLandmarker.createFromOptions(context, poseOptions)
        } catch (e: Exception) {
            Log.w(TAG, "GPU pose landmarker failed, falling back to CPU: ${e.message}")
            val cpuOpts = PoseLandmarker.PoseLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(POSE_MODEL)
                        .setDelegate(Delegate.CPU)
                        .build()
                )
                .setRunningMode(RunningMode.VIDEO)
                .setNumPoses(1)
                .setMinPoseDetectionConfidence(0.5f)
                .setMinPosePresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()
            PoseLandmarker.createFromOptions(context, cpuOpts)
        }

        // ── Hand Landmarker ──
        val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(
                BaseOptions.builder()
                    .setModelAssetPath(HAND_MODEL)
                    .setDelegate(Delegate.GPU)
                    .build()
            )
            .setRunningMode(RunningMode.VIDEO)
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .build()

        handLandmarker = try {
            HandLandmarker.createFromOptions(context, handOptions)
        } catch (e: Exception) {
            Log.w(TAG, "GPU hand landmarker failed, falling back to CPU: ${e.message}")
            val cpuOpts = HandLandmarker.HandLandmarkerOptions.builder()
                .setBaseOptions(
                    BaseOptions.builder()
                        .setModelAssetPath(HAND_MODEL)
                        .setDelegate(Delegate.CPU)
                        .build()
                )
                .setRunningMode(RunningMode.VIDEO)
                .setNumHands(2)
                .setMinHandDetectionConfidence(0.5f)
                .setMinHandPresenceConfidence(0.5f)
                .setMinTrackingConfidence(0.5f)
                .build()
            HandLandmarker.createFromOptions(context, cpuOpts)
        }

        Log.i(TAG, "MediaPipe landmarkers initialized")
    }

    /**
     * Extract 258 keypoints from a camera frame.
     *
     * @param bitmap The camera frame as ARGB_8888 bitmap
     * @param timestampMs Frame timestamp in milliseconds
     * @return FloatArray of size 258, or null if no pose detected
     */
    fun extract(bitmap: Bitmap, timestampMs: Long): FloatArray? {
        val mpImage = BitmapImageBuilder(bitmap).build()

        // Run pose detection
        val poseResult: PoseLandmarkerResult = poseLandmarker.detectForVideo(mpImage, timestampMs)
        if (poseResult.landmarks().isEmpty()) return null

        // Run hand detection
        val handResult: HandLandmarkerResult = handLandmarker.detectForVideo(mpImage, timestampMs)

        val keypoints = FloatArray(258)

        // ── Fill pose: 33 landmarks × 4 ──
        val poseLandmarks = poseResult.landmarks()[0]
        for (i in 0 until 33) {
            val lm = poseLandmarks[i]
            val base = i * 4
            keypoints[base] = lm.x()
            keypoints[base + 1] = lm.y()
            keypoints[base + 2] = lm.z()
            // Use world landmarks visibility if available, else 1.0
            keypoints[base + 3] = if (poseResult.worldLandmarks().isNotEmpty()) {
                poseResult.worldLandmarks()[0][i].visibility().orElse(1.0f)
            } else {
                1.0f
            }
        }

        // ── Fill hands: classify as left/right ──
        if (handResult.landmarks().isNotEmpty()) {
            val handedness = handResult.handednesses()
            for (h in handResult.landmarks().indices) {
                val hand = handResult.landmarks()[h]
                // MediaPipe handedness: "Left" means the hand appears on left of image
                // which corresponds to user's right hand (mirror), but we keep
                // consistent with model training — first label is the classification
                val label = if (h < handedness.size && handedness[h].isNotEmpty()) {
                    handedness[h][0].categoryName()
                } else {
                    if (h == 0) "Left" else "Right"
                }

                val offset = if (label == "Left") 132 else 195
                for (i in 0 until 21) {
                    val lm = hand[i]
                    val base = offset + i * 3
                    keypoints[base] = lm.x()
                    keypoints[base + 1] = lm.y()
                    keypoints[base + 2] = lm.z()
                }
            }
        }

        return keypoints
    }

    fun close() {
        poseLandmarker.close()
        handLandmarker.close()
    }
}
