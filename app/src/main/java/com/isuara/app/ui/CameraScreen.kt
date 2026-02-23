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
import androidx.camera.camera2.interop.Camera2Interop
import androidx.camera.camera2.interop.ExperimentalCamera2Interop

private const val TAG = "CameraScreen"

@androidx.annotation.OptIn(ExperimentalCamera2Interop::class)
@OptIn(ExperimentalMaterial3Api::class, androidx.camera.camera2.interop.ExperimentalCamera2Interop::class)
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

                // 1. Force the Preview to a lighter resolution and 60 FPS
                val previewBuilder = Preview.Builder()

                // Keep the preview resolution reasonable so it doesn't throttle the sensor
                // (This doesn't make the screen look bad, it just stops it from forcing 4K)
                previewBuilder.setTargetResolution(android.util.Size(640, 480))

                val previewExt = androidx.camera.camera2.interop.Camera2Interop.Extender(previewBuilder)
                previewExt.setCaptureRequestOption(
                    android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                    android.util.Range(30, 60)
                )

                val preview = previewBuilder.build().also {
                    it.surfaceProvider = previewView.surfaceProvider
                }

                val analysisBuilder = ImageAnalysis.Builder()
                    .setTargetResolution(android.util.Size(480, 360))
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)

                // Force camera sensor to output 45-60 FPS
                val ext = androidx.camera.camera2.interop.Camera2Interop.Extender(analysisBuilder)
                ext.setCaptureRequestOption(
                    android.hardware.camera2.CaptureRequest.CONTROL_AE_TARGET_FPS_RANGE,
                    android.util.Range(30, 60)
                )

                val imageAnalysis = analysisBuilder.build()

                val isFront = lensFacing == CameraSelector.LENS_FACING_FRONT

                var reusedBitmap: Bitmap? = null
                val canvas = android.graphics.Canvas()
                val matrix = android.graphics.Matrix()

                imageAnalysis.setAnalyzer(mlExecutor) { imageProxy ->
                    try {
                        val rawBitmap = imageProxy.toBitmap()
                        val rotation = imageProxy.imageInfo.rotationDegrees

                        // 1. Calculate the actual dimensions AFTER rotation (portrait swaps width/height)
                        val isPortrait = rotation == 90 || rotation == 270
                        val targetWidth = if (isPortrait) rawBitmap.height else rawBitmap.width
                        val targetHeight = if (isPortrait) rawBitmap.width else rawBitmap.height

                        // 2. Create the reusable bitmap ONCE with the CORRECT rotated dimensions
                        if (reusedBitmap == null || reusedBitmap!!.width != targetWidth) {
                            reusedBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
                        }

                        // 3. Set up rotation and mirroring properly CENTERED
                        matrix.reset()

                        // Move image center to origin (0,0)
                        matrix.postTranslate(-rawBitmap.width / 2f, -rawBitmap.height / 2f)

                        // Rotate it
                        matrix.postRotate(rotation.toFloat())

                        // Mirror if front camera
                        if (isFront) {
                            matrix.postScale(-1f, 1f)
                        }

                        // Move it back to the center of the new canvas
                        matrix.postTranslate(targetWidth / 2f, targetHeight / 2f)

                        // 4. Draw the raw camera feed onto our reusable bitmap
                        canvas.setBitmap(reusedBitmap)
                        canvas.drawColor(android.graphics.Color.BLACK, android.graphics.PorterDuff.Mode.CLEAR) // Clear previous frame
                        canvas.drawBitmap(rawBitmap, matrix, null)

                        // 5. Pass the correctly oriented bitmap to the AI
                        signPredictor.processFrame(reusedBitmap!!, imageProxy.imageInfo.timestamp / 1_000_000, isFront)

                    } finally {
                        imageProxy.close()
                    }

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
                // 3. Helper function to perfectly map MediaPipe [0..1] values to the screen
                fun mapX(xNorm: Float): Float {
                    return if (lensFacing == CameraSelector.LENS_FACING_FRONT) {
                        // Front camera: AI is mirrored, screen is mirrored. Perfect match!
                        (xNorm * scaledWidth) + offsetX
                    } else {
                        // Rear camera: AI is now mathematically mirrored, but the screen is normal.
                        // Flip the dots back so they line up with the real hand!
                        ((1f - xNorm) * scaledWidth) + offsetX
                    }
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