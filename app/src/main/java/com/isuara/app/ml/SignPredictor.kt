package com.isuara.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
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
import java.util.concurrent.atomic.AtomicInteger

class SignPredictor(context: Context) {

    data class PredictionState(
        val currentWord: String = "",
        val confidence: Float = 0f,
        val isConfident: Boolean = false,
        val bufferProgress: Float = 0f,
        val sentence: List<String> = emptyList(),
        val keypoints: FloatArray? = null,
        val imageWidth: Int = 480,  // PINPOINT: Fixes "No parameter found"
        val imageHeight: Int = 640  // PINPOINT: Fixes "No parameter found"
    )

    private val landmarkExtractor = LandmarkExtractor(context, this::onLandmarksExtracted)
    private val signInterpreter = SignInterpreter(context)
    private val labels: List<String>
    private val inferenceScope = CoroutineScope(Dispatchers.Default + SupervisorJob())
    private val isPredicting = AtomicBoolean(false)
    private val cooldownCounter = AtomicInteger(0)
    private val frameBuffer = ArrayDeque<FloatArray>(31)
    private val sentenceWords = mutableListOf<String>()
    private var lastWord = ""

    private val _state = MutableStateFlow(PredictionState())
    val state: StateFlow<PredictionState> = _state.asStateFlow()

    init {
        val jsonStr = context.assets.open("label_map.json").bufferedReader().use { it.readText() }
        labels = JSONObject(jsonStr).getJSONArray("actions_ordered").let { arr ->
            (0 until arr.length()).map { arr.getString(it) }
        }
    }

    fun processFrame(bitmap: Bitmap, timestampMs: Long) {
        // Update dimensions immediately for the UI mapping
        _state.update { it.copy(imageWidth = bitmap.width, imageHeight = bitmap.height) }
        landmarkExtractor.extractAsync(bitmap, timestampMs)
    }

    private fun onLandmarksExtracted(rawKeypoints: FloatArray?, timestampMs: Long) {
        _state.update { it.copy(keypoints = rawKeypoints) }
        if (rawKeypoints == null) return

        val normalized = FrameNormalizer.normalizeSingleFrame(rawKeypoints)
        synchronized(frameBuffer) {
            frameBuffer.addLast(normalized)
            if (frameBuffer.size > 30) frameBuffer.removeFirst()

            if (frameBuffer.size == 30 && cooldownCounter.get() <= 0 && isPredicting.compareAndSet(false, true)) {
                val sequence = frameBuffer.toTypedArray()
                inferenceScope.launch {
                    try {
                        val features = FrameNormalizer.buildSequenceFeatures(sequence)
                        val (idx, conf) = signInterpreter.predictTopClass(features)
                        updatePrediction(labels[idx], conf)
                    } finally {
                        isPredicting.set(false)
                    }
                }
            } else if (cooldownCounter.get() > 0) {
                cooldownCounter.decrementAndGet()
            }
            updateProgress()
        }
    }

    private fun updatePrediction(word: String, confidence: Float) {
        val isConfident = confidence >= 0.6f
        if (isConfident && word != lastWord && word != "Idle") {
            sentenceWords.add(word)
            lastWord = word
            if (sentenceWords.size > 8) sentenceWords.removeAt(0)
        }
        _state.update { it.copy(
            currentWord = if (isConfident) word else "$word?",
            confidence = confidence,
            isConfident = isConfident,
            sentence = sentenceWords.toList(),
            bufferProgress = 1f
        ) }
        cooldownCounter.set(if (isConfident) 10 else 5)
    }

    private fun updateProgress() {
        val progress = synchronized(frameBuffer) { frameBuffer.size.toFloat() / 30 }
        _state.update { it.copy(bufferProgress = progress) }
    }

    fun getSentenceWords() = sentenceWords.toList()
    fun resetAll() {
        synchronized(frameBuffer) { frameBuffer.clear() }
        sentenceWords.clear()
        lastWord = ""
        _state.update { PredictionState() }
    }
    fun close() {
        landmarkExtractor.close()
        signInterpreter.close()
    }
}