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

// Make sure to add this import at the top of your file:
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
    private val predictionStride = 2 // Predict every 2 frames (~25Hz at 50fps)
    private var frameCounter = 0

    private val frameBuffer = ArrayDeque<FloatArray>(31)
    private var previousFrame: FloatArray? = null

    // --- NEW: Re-add sentence list and initialize Segmenter ---
    private val sentenceWords = mutableListOf<String>()

    private val segmenter = SignSegmenter(
        onEmit = { emittedWord ->
            // When the segmenter confirms a word, add it to our sentence!
            sentenceWords.add(emittedWord)
            if (sentenceWords.size > 8) sentenceWords.removeAt(0)

            _state.update { it.copy(sentence = sentenceWords.toList()) }
        },
        onTrackingUpdate = { word, confidence, isConfident ->
            // Update the UI smoothly
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

        if (rawKeypoints == null) {
            previousFrame = null
            return
        }

        val rawNormalized = FrameNormalizer.normalizeSingleFrame(rawKeypoints)
        val smoothedNormalized = FloatArray(rawNormalized.size)
        val prev = previousFrame

        // --- APPLY EMA SMOOTHING FILTER ---
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

            // --- Continuous Sliding Window Logic ---
            frameCounter = (frameCounter + 1) % predictionStride

            if (frameBuffer.size == 30 && frameCounter == 0) {
                readyToPredict = true
                snapshot = frameBuffer.toTypedArray()
            }
            updateProgress()
        }

        // Launch inference using a Drop-If-Busy lock
        if (readyToPredict && snapshot != null) {
            // Only enter if false, and immediately set to true
            if (isPredicting.compareAndSet(false, true)) {
                inferenceScope.launch {
                    try {
                        val features = FrameNormalizer.buildSequenceFeatures(snapshot!!)
                        val (idx, conf) = signInterpreter.predictTopClass(features)
                        val predictedClass = labels[idx]

                        segmenter.processPrediction(predictedClass, conf)
                    } finally {
                        // ALWAYS release the lock, even if the model throws an error
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
        sentenceWords.clear()
        segmenter.reset()
        _state.update { PredictionState() }
    }

    // Also re-add this getter if your UI needs it:
    fun getSentenceWords() = sentenceWords.toList()

    fun close() {
        landmarkExtractor.close()
        signInterpreter.close()
    }
}