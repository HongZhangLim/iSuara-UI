package com.isuara.app.ml

class SignSegmenter(
    private val onEmit: (String) -> Unit,
    private val onTrackingUpdate: (word: String, confidence: Float, isConfident: Boolean) -> Unit
) {
    private enum class State {
        IDLE,
        TRACKING,
        DEBOUNCING
    }

    // --- Configuration Thresholds ---
    private val minOnset = 2
    private val minHold = 3
    private val fastCommitCount = 2
    private val debounceLimit = 3

    // NEW: How many "Idle" or low-confidence frames we need to see
    // before we allow the user to sign the EXACT SAME word again.
    // At 25Hz, 5 frames is ~200ms.
    private val framesToClearEmission = 5

    private val confidenceThreshold = 0.5f
    private val highConfidenceThreshold = 0.85f

    // --- State Variables ---
    private var currentState = State.IDLE
    private var trackedClass = ""
    private var trackingCount = 0
    private var highConfidenceCount = 0
    private var debounceCounter = 0

    // NEW: Memory locks to prevent spamming
    private var lastEmittedClass = ""
    private var idleCount = 0

    fun processPrediction(predictedClass: String, confidence: Float) {
        // 1. Handle DEBOUNCING (post-emission cooldown)
        if (currentState == State.DEBOUNCING) {
            debounceCounter--
            if (debounceCounter <= 0) {
                resetToIdle()
            } else {
                onTrackingUpdate("", 0f, false)
            }
            return
        }

        // 2. Handle IDLE / Low Confidence / Background noise
        val isIdleClass = predictedClass.equals("idle", ignoreCase = true) ||
                predictedClass.equals("background", ignoreCase = true)

        if (isIdleClass || confidence < confidenceThreshold) {
            // Increment our idle tracker
            idleCount++

            // If the user drops their hands for long enough, clear the lock
            // so they can sign the same word again if they want to.
            if (idleCount >= framesToClearEmission) {
                lastEmittedClass = ""
            }

            if (currentState != State.IDLE) {
                resetToIdle()
            } else {
                onTrackingUpdate("", 0f, false)
            }
            return
        }

        // 3. Process Valid Predictions
        idleCount = 0 // Reset idle counter because we see a valid sign

        // NEW: PREVENT DUPLICATES
        if (predictedClass == lastEmittedClass) {
            // The user is still holding the sign they JUST emitted.
            // We do NOT want to track it for a second emission.
            // But we DO want the UI to keep showing it as a solid, confident word on screen.
            onTrackingUpdate(predictedClass, confidence, true)
            return
        }

        when (currentState) {
            State.IDLE -> {
                if (predictedClass == trackedClass) {
                    trackingCount++
                    updateHighConfidence(confidence)

                    if (trackingCount >= minOnset) {
                        currentState = State.TRACKING
                        onTrackingUpdate(trackedClass, confidence, false)
                    }
                } else {
                    trackedClass = predictedClass
                    trackingCount = 1
                    highConfidenceCount = 0
                    updateHighConfidence(confidence)
                }
            }

            State.TRACKING -> {
                if (predictedClass == trackedClass) {
                    trackingCount++
                    updateHighConfidence(confidence)

                    val isConfident = trackingCount >= minOnset
                    onTrackingUpdate(trackedClass, confidence, isConfident)

                    if (trackingCount >= minHold || highConfidenceCount >= fastCommitCount) {
                        emitAndDebounce()
                    }
                } else {
                    trackedClass = predictedClass
                    trackingCount = 1
                    highConfidenceCount = 0
                    updateHighConfidence(confidence)
                    currentState = State.IDLE
                    onTrackingUpdate("", 0f, false)
                }
            }

            State.DEBOUNCING -> { }
        }
    }

    private fun updateHighConfidence(confidence: Float) {
        if (confidence >= highConfidenceThreshold) {
            highConfidenceCount++
        } else {
            highConfidenceCount = 0
        }
    }

    private fun emitAndDebounce() {
        onEmit(trackedClass)

        // NEW: Lock this word so it can't be emitted again until hands are dropped
        lastEmittedClass = trackedClass

        currentState = State.DEBOUNCING
        debounceCounter = debounceLimit
        onTrackingUpdate("", 0f, false)
    }

    private fun resetToIdle() {
        currentState = State.IDLE
        trackedClass = ""
        trackingCount = 0
        highConfidenceCount = 0
        onTrackingUpdate("", 0f, false)
    }

    fun reset() {
        resetToIdle()
        debounceCounter = 0
        lastEmittedClass = ""
        idleCount = 0
    }
}