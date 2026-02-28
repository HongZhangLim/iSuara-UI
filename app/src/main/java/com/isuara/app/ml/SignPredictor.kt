package com.isuara.app.ml

import android.content.Context
import android.graphics.Bitmap
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import org.json.JSONObject
import java.util.concurrent.atomic.AtomicBoolean

class SignPredictor(context: Context) {

    data class PredictionState(
        val currentWord: String = "",
        val confidence: Float = 0f,
        val isConfident: Boolean = false,
        val bufferProgress: Float = 0f,
        val sentence: List<String> = emptyList(),
        val keypoints: FloatArray? = null,
        val imageWidth: Int = 480,
        val imageHeight: Int = 640
    )

    private val landmarkExtractor = LandmarkExtractor(context, this::onLandmarksExtracted)
    private val signInterpreter = SignInterpreter(context)
    private val labels: List<String>
    private val inferenceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())

    private val isPredicting = AtomicBoolean(false)

    // --- Stride logic ---
    private val predictionStride = 2
    private var frameCounter = 0

    private val frameBuffer = ArrayDeque<FloatArray>(31)

    // Smooths the input coordinates
    private var previousFrame: FloatArray? = null

    // Smooths the output probabilities (NEW)
    private var previousPredictions: FloatArray? = null

    private val sentenceWords = mutableListOf<String>()

    private val segmenter = SignSegmenter(
        onEmit = { emittedWord ->
            sentenceWords.add(emittedWord)
            if (sentenceWords.size > 8) sentenceWords.removeAt(0)
            _state.update { it.copy(sentence = sentenceWords.toList()) }
        },
        onTrackingUpdate = { word, confidence, isConfident ->
            _state.update { it.copy(
                currentWord = if (isConfident || word.isEmpty()) word else "$word?",
                confidence = confidence,
                isConfident = isConfident
            ) }
        }
    )

    private val _state = MutableStateFlow(PredictionState())
    val state: StateFlow<PredictionState> = _state.asStateFlow()

    init {
        val jsonStr = context.assets.open("label_map.json").bufferedReader().use { it.readText() }
        labels = JSONObject(jsonStr).getJSONArray("actions_ordered").let { arr ->
            (0 until arr.length()).map { arr.getString(it) }
        }
    }

    fun processFrame(bitmap: Bitmap, timestampMs: Long, isFrontCamera: Boolean = true) {
        _state.update { it.copy(imageWidth = bitmap.width, imageHeight = bitmap.height) }
        landmarkExtractor.extractAsync(bitmap, timestampMs, isFrontCamera)
    }

    private fun onLandmarksExtracted(rawKeypoints: FloatArray?, timestampMs: Long) {
        _state.update { it.copy(keypoints = rawKeypoints) }

        // --- THE VANISHING HANDS FIX ---
        if (rawKeypoints == null) {
            // 1. Reset EMA smoothers so old data doesn't bleed into new signs
            previousFrame = null
            previousPredictions = null

            // 2. Clear the physical frame buffer so we don't hold "stale" poses
            synchronized(frameBuffer) {
                frameBuffer.clear()
            }
            updateProgress() // Keep UI buffer progress accurate (it will drop to 0)

            // 3. Manually feed an "idle" prediction to the segmenter to increment idleCount
            // and unlock the lastEmittedClass lock!
            segmenter.processPrediction("idle", 1.0f)

            return
        }
        // --------------------------------

        val rawNormalized = FrameNormalizer.normalizeSingleFrame(rawKeypoints)
        val smoothedNormalized = FloatArray(rawNormalized.size)
        val prev = previousFrame

        // 1. INPUT EMA SMOOTHING
        if (prev == null) {
            System.arraycopy(rawNormalized, 0, smoothedNormalized, 0, rawNormalized.size)
        } else {
            val alpha = 0.4f
            for (i in rawNormalized.indices) {
                smoothedNormalized[i] = (rawNormalized[i] * alpha) + (prev[i] * (1f - alpha))
            }
        }
        previousFrame = smoothedNormalized.clone()

        var readyToPredict = false
        var snapshot: Array<FloatArray>? = null

        synchronized(frameBuffer) {
            frameBuffer.addLast(smoothedNormalized)
            if (frameBuffer.size > 30) frameBuffer.removeFirst()

            frameCounter = (frameCounter + 1) % predictionStride

            if (frameBuffer.size == 30 && frameCounter == 0) {
                readyToPredict = true
                snapshot = frameBuffer.toTypedArray()
            }
            updateProgress()
        }

        if (readyToPredict && snapshot != null) {
            if (isPredicting.compareAndSet(false, true)) {
                inferenceScope.launch {
                    try {
                        val features = FrameNormalizer.buildSequenceFeatures(snapshot!!)

                        val rawPredictions = signInterpreter.predict(features)

                        // 2. OUTPUT EMA SMOOTHING
                        val smoothedPredictions = FloatArray(rawPredictions.size)
                        val prevPreds = previousPredictions

                        if (prevPreds == null) {
                            System.arraycopy(rawPredictions, 0, smoothedPredictions, 0, rawPredictions.size)
                        } else {
                            val alpha = 0.4f
                            for (i in rawPredictions.indices) {
                                smoothedPredictions[i] = (rawPredictions[i] * alpha) + (prevPreds[i] * (1f - alpha))
                            }
                        }
                        previousPredictions = smoothedPredictions.clone()

                        // 3. Find the NEW top class from the smoothed array
                        var topIdx = -1
                        var maxConf = -1f
                        for (i in smoothedPredictions.indices) {
                            if (smoothedPredictions[i] > maxConf) {
                                maxConf = smoothedPredictions[i]
                                topIdx = i
                            }
                        }

                        val predictedClass = labels[topIdx]
                        segmenter.processPrediction(predictedClass, maxConf)

                    } finally {
                        isPredicting.set(false)
                    }
                }
            }
        }
    }

    private fun updateProgress() {
        val progress = synchronized(frameBuffer) { frameBuffer.size.toFloat() / 30 }
        _state.update { it.copy(bufferProgress = progress) }
    }

    fun resetAll() {
        synchronized(frameBuffer) { frameBuffer.clear() }
        previousFrame = null
        previousPredictions = null // Reset prediction EMA
        sentenceWords.clear()
        segmenter.reset()
        _state.update { PredictionState() }
    }

    fun getSentenceWords() = sentenceWords.toList()

    fun close() {
        landmarkExtractor.close()
        signInterpreter.close()
    }
}