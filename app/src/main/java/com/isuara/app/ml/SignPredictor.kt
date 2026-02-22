package com.isuara.app.ml

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import org.json.JSONObject

/**
 * SignPredictor — orchestrates the full ML pipeline:
 *   CameraFrame → LandmarkExtractor → FrameNormalizer → SignInterpreter
 *
 * Manages sliding window of 30 frames, cooldown, sentence buffer.
 * All processing runs on the caller's thread (CameraX executor).
 */
class SignPredictor(context: Context) {

    companion object {
        private const val TAG = "SignPredictor"
        private const val CONFIDENCE_THRESHOLD = 0.6f
        private const val COOLDOWN_FRAMES = 10
        private const val MAX_SENTENCE_WORDS = 8
    }

    // ── ML components ──
    private val landmarkExtractor = LandmarkExtractor(context)
    private val signInterpreter = SignInterpreter(context)
    private val labels: List<String>

    // ── Sliding window state ──
    private val frameBuffer = ArrayDeque<FloatArray>(FrameNormalizer.SEQUENCE_LENGTH + 1)
    private var cooldownCounter = 0

    // ── Sentence state ──
    private val sentenceWords = mutableListOf<String>()
    private var lastWord = ""

    // ── Observable state for UI ──
    data class PredictionState(
        val currentWord: String = "",
        val confidence: Float = 0f,
        val isConfident: Boolean = false,
        val bufferProgress: Float = 0f,
        val sentence: List<String> = emptyList()
    )

    private val _state = MutableStateFlow(PredictionState())
    val state: StateFlow<PredictionState> = _state.asStateFlow()

    init {
        // Load label map
        val jsonStr = context.assets.open("label_map.json")
            .bufferedReader().use { it.readText() }
        val json = JSONObject(jsonStr)
        val arr = json.getJSONArray("actions_ordered")
        labels = (0 until arr.length()).map { arr.getString(it) }
        Log.i(TAG, "Loaded ${labels.size} labels")
    }

    /**
     * Process one camera frame. Call from CameraX analysis thread.
     *
     * @param bitmap ARGB_8888 camera frame
     * @param timestampMs frame timestamp in ms
     */
    fun processFrame(bitmap: Bitmap, timestampMs: Long) {
        // ── Stage 0: Extract landmarks ──
        val rawKeypoints = landmarkExtractor.extract(bitmap, timestampMs)
        if (rawKeypoints == null) {
            updateProgress()
            return
        }

        // ── Stages 1+2: Anchor + Scale normalization ──
        val normalized = FrameNormalizer.normalizeSingleFrame(rawKeypoints)
        frameBuffer.addLast(normalized)
        if (frameBuffer.size > FrameNormalizer.SEQUENCE_LENGTH) {
            frameBuffer.removeFirst()
        }

        // ── Predict when buffer full and not in cooldown ──
        if (frameBuffer.size == FrameNormalizer.SEQUENCE_LENGTH && cooldownCounter <= 0) {
            // Stages 3+4+5: Velocity, Acceleration, Engineered
            val sequence = frameBuffer.toTypedArray()
            val features = FrameNormalizer.buildSequenceFeatures(sequence)

            // Stage 6 (z-score) is baked into the model
            val (predIdx, confidence) = signInterpreter.predictTopClass(features)

            if (confidence >= CONFIDENCE_THRESHOLD) {
                val word = labels[predIdx]

                // Add to sentence (skip consecutive duplicates and Idle)
                if (word != lastWord && word != "Idle") {
                    sentenceWords.add(word)
                    lastWord = word
                    if (sentenceWords.size > MAX_SENTENCE_WORDS) {
                        sentenceWords.removeAt(0)
                    }
                }

                _state.value = PredictionState(
                    currentWord = word,
                    confidence = confidence,
                    isConfident = true,
                    bufferProgress = 1f,
                    sentence = sentenceWords.toList()
                )
                cooldownCounter = COOLDOWN_FRAMES
            } else {
                _state.value = PredictionState(
                    currentWord = "${labels[predIdx]}?",
                    confidence = confidence,
                    isConfident = false,
                    bufferProgress = 1f,
                    sentence = sentenceWords.toList()
                )
                cooldownCounter = COOLDOWN_FRAMES / 2
            }
        } else {
            if (cooldownCounter > 0) cooldownCounter--
            updateProgress()
        }
    }

    private fun updateProgress() {
        val prev = _state.value
        val progress = frameBuffer.size.toFloat() / FrameNormalizer.SEQUENCE_LENGTH
        if (progress != prev.bufferProgress) {
            _state.value = prev.copy(bufferProgress = progress)
        }
    }

    fun getSentenceWords(): List<String> = sentenceWords.toList()

    fun resetBuffer() {
        frameBuffer.clear()
        cooldownCounter = 0
        _state.value = _state.value.copy(
            currentWord = "",
            confidence = 0f,
            bufferProgress = 0f
        )
    }

    fun resetSentence() {
        sentenceWords.clear()
        lastWord = ""
        _state.value = _state.value.copy(sentence = emptyList())
    }

    fun resetAll() {
        resetBuffer()
        resetSentence()
    }

    fun close() {
        landmarkExtractor.close()
        signInterpreter.close()
    }
}
