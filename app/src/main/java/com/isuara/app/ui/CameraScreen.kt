package com.isuara.app.ui

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import com.isuara.app.ml.SignPredictor
import com.isuara.app.service.GeminiTranslator
import com.isuara.app.service.TtsService
import kotlinx.coroutines.launch
import java.util.concurrent.Executors

private const val TAG = "CameraScreen"

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CameraScreen(
    signPredictor: SignPredictor,
    geminiTranslator: GeminiTranslator?,
    ttsService: TtsService
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val scope = rememberCoroutineScope()

    // ── Observe ML state ──
    val predictionState by signPredictor.state.collectAsState()

    // ── Local UI state ──
    var translatedText by remember { mutableStateOf("") }
    var isTranslating by remember { mutableStateOf(false) }
    var fpsCounter by remember { mutableIntStateOf(0) }
    var displayFps by remember { mutableIntStateOf(0) }
    var lastFpsTime by remember { mutableLongStateOf(System.currentTimeMillis()) }

    // ── Camera executor (single thread for ML) ──
    val mlExecutor = remember { Executors.newSingleThreadExecutor() }

    Box(modifier = Modifier.fillMaxSize()) {
        // ══════════════════════════════════════════
        // Camera Preview (full screen, behind everything)
        // ══════════════════════════════════════════
        AndroidView(
            factory = { ctx ->
                val previewView = PreviewView(ctx).apply {
                    scaleType = PreviewView.ScaleType.FILL_CENTER
                    implementationMode = PreviewView.ImplementationMode.PERFORMANCE
                }

                val cameraProviderFuture = ProcessCameraProvider.getInstance(ctx)
                cameraProviderFuture.addListener({
                    val cameraProvider = cameraProviderFuture.get()

                    val preview = Preview.Builder()
                        .build()
                        .also { it.surfaceProvider = previewView.surfaceProvider }

                    val imageAnalysis = ImageAnalysis.Builder()
                        .setTargetResolution(android.util.Size(640, 480))
                        .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                        .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                        .build()

                    imageAnalysis.setAnalyzer(mlExecutor) { imageProxy ->
                        processImageProxy(imageProxy, signPredictor)

                        // FPS tracking
                        fpsCounter++
                        val now = System.currentTimeMillis()
                        if (now - lastFpsTime >= 1000) {
                            displayFps = fpsCounter
                            fpsCounter = 0
                            lastFpsTime = now
                        }
                    }

                    try {
                        cameraProvider.unbindAll()
                        cameraProvider.bindToLifecycle(
                            lifecycleOwner,
                            CameraSelector.DEFAULT_FRONT_CAMERA,
                            preview,
                            imageAnalysis
                        )
                    } catch (e: Exception) {
                        Log.e(TAG, "Camera bind failed", e)
                    }
                }, ContextCompat.getMainExecutor(ctx))

                previewView
            },
            modifier = Modifier.fillMaxSize()
        )

        // ══════════════════════════════════════════
        // Top overlay: current prediction + confidence
        // ══════════════════════════════════════════
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .background(Color.Black.copy(alpha = 0.5f))
                .padding(horizontal = 16.dp, vertical = 12.dp)
                .statusBarsPadding()
        ) {
            // FPS + buffer progress
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text(
                    text = "iSuara",
                    color = Color.White,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )

                Text(
                    text = "${displayFps} FPS",
                    color = Color.White.copy(alpha = 0.7f),
                    fontSize = 12.sp
                )
            }

            // Buffer progress bar
            Spacer(modifier = Modifier.height(6.dp))
            LinearProgressIndicator(
                progress = { predictionState.bufferProgress },
                modifier = Modifier
                    .fillMaxWidth()
                    .height(3.dp)
                    .clip(RoundedCornerShape(2.dp)),
                color = if (predictionState.bufferProgress >= 1f) Color(0xFF4CAF50) else Color(0xFF2196F3),
                trackColor = Color.White.copy(alpha = 0.2f),
            )

            // Current prediction
            Spacer(modifier = Modifier.height(8.dp))
            AnimatedVisibility(
                visible = predictionState.currentWord.isNotEmpty(),
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                Row(
                    verticalAlignment = Alignment.Bottom,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text(
                        text = predictionState.currentWord,
                        color = if (predictionState.isConfident) Color(0xFF4CAF50) else Color(0xFFFF9800),
                        fontSize = 28.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = "${(predictionState.confidence * 100).toInt()}%",
                        color = Color.White.copy(alpha = 0.6f),
                        fontSize = 14.sp
                    )
                }
            }
        }

        // ══════════════════════════════════════════
        // Bottom panel: sentence + translate + TTS
        // ══════════════════════════════════════════
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .background(Color.Black.copy(alpha = 0.65f))
                .navigationBarsPadding()
                .padding(horizontal = 16.dp, vertical = 12.dp)
        ) {
            // Sentence words
            if (predictionState.sentence.isNotEmpty()) {
                Text(
                    text = predictionState.sentence.joinToString("  "),
                    color = Color.White,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Medium,
                    modifier = Modifier.fillMaxWidth()
                )
                Spacer(modifier = Modifier.height(4.dp))
            }

            // Translated sentence
            AnimatedVisibility(visible = translatedText.isNotEmpty()) {
                Text(
                    text = translatedText,
                    color = Color(0xFF4CAF50),
                    fontSize = 16.sp,
                    fontWeight = FontWeight.SemiBold,
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF1B5E20).copy(alpha = 0.3f), RoundedCornerShape(6.dp))
                        .padding(8.dp)
                )
            }

            if (isTranslating) {
                Text(
                    text = "Translating...",
                    color = Color(0xFFFFEB3B),
                    fontSize = 14.sp,
                    modifier = Modifier.padding(vertical = 4.dp)
                )
            }

            Spacer(modifier = Modifier.height(8.dp))

            // Action buttons
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceEvenly,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Reset button
                FilledTonalButton(
                    onClick = {
                        signPredictor.resetAll()
                        translatedText = ""
                        isTranslating = false
                        ttsService.stop()
                    },
                    colors = ButtonDefaults.filledTonalButtonColors(
                        containerColor = Color.White.copy(alpha = 0.15f),
                        contentColor = Color.White
                    )
                ) {
                    Icon(Icons.Default.Refresh, contentDescription = "Reset", modifier = Modifier.size(18.dp))
                    Spacer(modifier = Modifier.width(4.dp))
                    Text("Reset", fontSize = 13.sp)
                }

                // Translate button
                Button(
                    onClick = {
                        val words = signPredictor.getSentenceWords()
                        if (words.isNotEmpty() && !isTranslating) {
                            isTranslating = true
                            translatedText = ""
                            scope.launch {
                                val result = geminiTranslator?.translate(words)
                                    ?: "[API key not configured]"
                                translatedText = result
                                isTranslating = false
                            }
                        }
                    },
                    enabled = predictionState.sentence.isNotEmpty() && !isTranslating,
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF2196F3)
                    )
                ) {
                    Text("Translate", fontSize = 13.sp)
                }

                // TTS button
                Button(
                    onClick = {
                        val textToSpeak = translatedText.ifEmpty {
                            predictionState.sentence.joinToString(" ")
                        }
                        if (textToSpeak.isNotEmpty()) {
                            ttsService.speak(textToSpeak)
                        }
                    },
                    enabled = translatedText.isNotEmpty() || predictionState.sentence.isNotEmpty(),
                    colors = ButtonDefaults.buttonColors(
                        containerColor = Color(0xFF4CAF50)
                    )
                ) {
                    Text("Speak", fontSize = 13.sp)
                }
            }
        }
    }
}

/**
 * Convert ImageProxy to Bitmap and feed to SignPredictor.
 * Runs on the ML executor thread.
 */
private fun processImageProxy(imageProxy: ImageProxy, signPredictor: SignPredictor) {
    try {
        val bitmap = imageProxy.toBitmap()

        // Front camera needs horizontal flip (mirror)
        val matrix = Matrix().apply { postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f) }
        val flipped = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        signPredictor.processFrame(flipped, imageProxy.imageInfo.timestamp / 1_000_000)

        if (flipped !== bitmap) flipped.recycle()
        bitmap.recycle()
    } catch (e: Exception) {
        Log.e(TAG, "Frame processing error", e)
    } finally {
        imageProxy.close()
    }
}
