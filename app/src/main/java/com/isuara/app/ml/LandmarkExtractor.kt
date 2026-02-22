package com.isuara.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.framework.image.MPImage
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.core.Delegate
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult
import java.util.concurrent.ConcurrentHashMap

class LandmarkExtractor(
    context: Context,
    private val onResult: (FloatArray?, Long) -> Unit // PINPOINT: Fixes "Too many arguments"
) {

    companion object {
        private const val TAG = "LandmarkExtractor"
    }

    private var poseLandmarker: PoseLandmarker? = null
    private var handLandmarker: HandLandmarker? = null

    private class FrameResult {
        var poseDone = false
        var handDone = false
        var features: FloatArray? = null
        var hasData = false
    }
    private val pendingFrames = ConcurrentHashMap<Long, FrameResult>()

    init {
        // Use CPU for stability, LIVE_STREAM for 30+ FPS performance
        val poseOptions = PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker_lite.task").setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener(this::onPoseResult)
            .build()

        val handOptions = HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").setDelegate(Delegate.CPU).build())
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setNumHands(2)
            .setResultListener(this::onHandResult)
            .build()

        poseLandmarker = PoseLandmarker.createFromOptions(context, poseOptions)
        handLandmarker = HandLandmarker.createFromOptions(context, handOptions)
    }

    fun extractAsync(bitmap: Bitmap, timestampMs: Long) { // PINPOINT: Fixes "Unresolved reference"
        val image = BitmapImageBuilder(bitmap).build()
        pendingFrames[timestampMs] = FrameResult()
        poseLandmarker?.detectAsync(image, timestampMs)
        handLandmarker?.detectAsync(image, timestampMs)
    }

    private fun onPoseResult(result: PoseLandmarkerResult, image: MPImage) {
        val ts = result.timestampMs() // PINPOINT: Fixes private timestamp error
        val frame = pendingFrames[ts] ?: return
        synchronized(frame) {
            if (frame.features == null) frame.features = FloatArray(258)
            if (result.landmarks().isNotEmpty()) {
                frame.hasData = true
                val poseLms = result.landmarks()[0]
                for (i in 0 until 33) {
                    val idx = i * 4
                    frame.features!![idx] = poseLms[i].x()
                    frame.features!![idx + 1] = poseLms[i].y()
                    frame.features!![idx + 2] = poseLms[i].z()
                    frame.features!![idx + 3] = poseLms[i].visibility().orElse(0f)
                }
            }
            frame.poseDone = true
            checkCompletion(ts, frame)
        }
    }

    private fun onHandResult(result: HandLandmarkerResult, image: MPImage) {
        val ts = result.timestampMs() // PINPOINT: Fixes private timestamp error
        val frame = pendingFrames[ts] ?: return
        synchronized(frame) {
            if (frame.features == null) frame.features = FloatArray(258)
            if (result.landmarks().isNotEmpty()) {
                frame.hasData = true
                for (i in result.landmarks().indices) {
                    val isLeft = result.handednesses()[i][0].categoryName() == "Left"
                    val offset = if (isLeft) 132 else 195
                    for (j in 0 until 21) {
                        val idx = offset + (j * 3)
                        frame.features!![idx] = result.landmarks()[i][j].x()
                        frame.features!![idx+1] = result.landmarks()[i][j].y()
                        frame.features!![idx+2] = result.landmarks()[i][j].z()
                    }
                }
            }
            frame.handDone = true
            checkCompletion(ts, frame)
        }
    }

    private fun checkCompletion(ts: Long, frame: FrameResult) {
        if (frame.poseDone && frame.handDone) {
            pendingFrames.remove(ts)
            onResult(if (frame.hasData) frame.features else null, ts)
        }
    }

    fun close() {
        poseLandmarker?.close()
        handLandmarker?.close()
    }
}