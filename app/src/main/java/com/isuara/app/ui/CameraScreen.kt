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
import androidx.compose.ui.draw.scale
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
    var showLandmarks by remember { mutableStateOf(false) }
    var translatedText by remember { mutableStateOf("") }
    var isTranslating by remember { mutableStateOf(false) }
    var fpsCounter by remember { mutableIntStateOf(0) }
    var displayFps by remember { mutableIntStateOf(0) }
    var lastFpsTime by remember { mutableLongStateOf(System.currentTimeMillis()) }
    var lensFacing by remember { mutableIntStateOf(CameraSelector.LENS_FACING_FRONT) }

    // ── Camera executor (single thread for ML) ──
    val mlExecutor = remember { Executors.newSingleThreadExecutor() }

    Box(modifier = Modifier.fillMaxSize()) {
        // ══════════════════════════════════════════
        // Camera Preview Setup (LaunchedEffect + AndroidView)
        // ══════════════════════════════════════════

        // 1. Create the physical view exactly once
        val previewView = remember {
            androidx.camera.view.PreviewView(context).apply {
                scaleType = androidx.camera.view.PreviewView.ScaleType.FILL_CENTER
                implementationMode = androidx.camera.view.PreviewView.ImplementationMode.PERFORMANCE
            }
        }

        // 2. Re-bind the camera ONLY when 'lensFacing' changes
        LaunchedEffect(lensFacing) {
            val cameraProviderFuture = ProcessCameraProvider.getInstance(context)
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

                val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT

                imageAnalysis.setAnalyzer(mlExecutor) { imageProxy ->
                    processImageProxy(imageProxy, signPredictor, isFront)

                    // FPS tracking
                    fpsCounter++
                    val now = System.currentTimeMillis()
                    if (now - lastFpsTime >= 1000) {
                        displayFps = fpsCounter
                        fpsCounter = 0
                        lastFpsTime = now
                    }
                }

                val cameraSelector = CameraSelector.Builder().requireLensFacing(lensFacing).build()

                try {
                    cameraProvider.unbindAll() // Stop current camera
                    cameraProvider.bindToLifecycle(
                        lifecycleOwner,
                        cameraSelector,    // Start new camera (Front or Back)
                        preview,
                        imageAnalysis
                    )
                } catch (e: Exception) {
                    Log.e(TAG, "Camera bind failed", e)
                }
            }, ContextCompat.getMainExecutor(context))
        }

        // 3. Display the view on screen
        AndroidView(
            factory = { previewView },
            modifier = Modifier.fillMaxSize()
        )

        if (showLandmarks && predictionState.keypoints != null) {
            androidx.compose.foundation.Canvas(modifier = Modifier.fillMaxSize()) {
                val keypoints = predictionState.keypoints!!
                val radius = 6f

                // PINPOINT: These now reference the fields added to PredictionState
                val imageWidth = predictionState.imageWidth.toFloat()
                val imageHeight = predictionState.imageHeight.toFloat()

                // 1. Calculate how much the preview scaled the image to fill the screen
                val scale = java.lang.Math.max(size.width / imageWidth, size.height / imageHeight)
                val scaledWidth = imageWidth * scale
                val scaledHeight = imageHeight * scale

                // 2. Calculate the crop offsets (how much was chopped off the sides/top)
                val offsetX = (size.width - scaledWidth) / 2f
                val offsetY = (size.height - scaledHeight) / 2f

                // 3. Helper function to perfectly map MediaPipe [0..1] values to the screen
                fun mapX(xNorm: Float): Float {
                    // PINPOINT: Mirror the x-coordinate for the front camera
                    // PreviewView mirrors the display, so we must mirror the landmark mapping
                    return ((1f - xNorm) * scaledWidth) + offsetX
                }

                fun mapY(yNorm: Float) = (yNorm * scaledHeight) + offsetY

                for (i in 0 until 33) {
                    val x = mapX(keypoints[i * 4])
                    val y = mapY(keypoints[i * 4 + 1])

                    // PINPOINT: Use visibility threshold to prevent "ghost" dots
                    val visibility = keypoints[i * 4 + 3]
                    if (visibility > 0.5f) {
                        drawCircle(Color.Green, radius, androidx.compose.ui.geometry.Offset(x, y))
                    }
                }

                // 2. Left Hand (Magenta)
                for (i in 0 until 21) {
                    val idx = 132 + (i * 3)
                    val x = mapX(keypoints[idx])
                    val y = mapY(keypoints[idx + 1])
                    if (keypoints[idx] > 0f) drawCircle(Color.Magenta, radius, androidx.compose.ui.geometry.Offset(x, y))
                }

                // 3. Right Hand (Cyan)
                for (i in 0 until 21) {
                    val idx = 195 + (i * 3)
                    val x = mapX(keypoints[idx])
                    val y = mapY(keypoints[idx + 1])
                    if (keypoints[idx] > 0f) drawCircle(Color.Cyan, radius, androidx.compose.ui.geometry.Offset(x, y))
                }
            }
        }

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

                Row(verticalAlignment = Alignment.CenterVertically) {
                    IconButton(onClick = {
                        lensFacing = if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                            CameraSelector.LENS_FACING_BACK
                        } else {
                            CameraSelector.LENS_FACING_FRONT
                        }
                    }) {
                        Icon(
                            imageVector = Icons.Default.Refresh,
                            contentDescription = "Flip Camera",
                            tint = Color.White
                        )
                    }
                    Text("Dots", color = Color.White.copy(alpha = 0.8f), fontSize = 12.sp)
                    Switch(
                        checked = showLandmarks,
                        onCheckedChange = { showLandmarks = it },
                        modifier = Modifier.scale(0.7f) // Make switch smaller to fit top bar
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "${displayFps} FPS",
                        color = Color.White.copy(alpha = 0.7f),
                        fontSize = 12.sp
                    )
                }
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
private fun processImageProxy(imageProxy: ImageProxy, signPredictor: SignPredictor, isFrontCamera: Boolean) {
    try {
        val bitmap = imageProxy.toBitmap()
        val matrix = Matrix().apply {
            // PINPOINT: Rotate the frame so the AI sees a standing person
            postRotate(imageProxy.imageInfo.rotationDegrees.toFloat())
            if (isFrontCamera) {
                postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
            }
        }
        val processedBitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)

        signPredictor.processFrame(processedBitmap, imageProxy.imageInfo.timestamp / 1_000_000)

        if (processedBitmap !== bitmap) processedBitmap.recycle()
        bitmap.recycle()
    } finally {
        imageProxy.close()
    }
}
