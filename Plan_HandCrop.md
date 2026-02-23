Here is the full, comprehensive plan for implementing the dynamic hand-cropping strategy (Step 4) in your Android application.

This plan involves splitting the hand landmarker into two separate instances (to fix hand-entanglement bugs), using the pose data to dynamically size and center a crop box, and remapping the coordinates back to the full frame.

### Phase 1: Update State Tracking and Frame Synchronization

To make cropping work, you must save the wrist coordinates from the *previous* frame's Pose detection to use as the center points for the *current* frame's Hand detection.

1. **Add Pose Tracking Variables:** Inside `LandmarkExtractor.kt`, add variables to store the last known normalized coordinates of the wrists and the dynamic crop size.
```kotlin
private var lastLeftWristNormX = -1f
private var lastLeftWristNormY = -1f
private var lastRightWristNormX = -1f
private var lastRightWristNormY = -1f
private var lastDynamicCropSize = 200 // Default fallback size in pixels

```


2. **Update `FrameResult`:** You need to track the crop bounding boxes so the async callbacks know how to reverse the math later. You also need to track the left and right hands independently.
```kotlin
private class FrameResult {
    var poseDone = false
    var leftHandDone = false
    var rightHandDone = false
    var features: FloatArray? = null
    var hasData = false
    var isFrontCamera = true

    // Track the crop boxes used for this specific frame
    var leftCropStartX = 0
    var leftCropStartY = 0
    var leftCropSize = 0

    var rightCropStartX = 0
    var rightCropStartY = 0
    var rightCropSize = 0
}

```



### Phase 2: Split the HandLandmarker

Since overlapping hands confuse MediaPipe, running two isolated landmarker instances on two separate wrist-cropped images is vastly superior.

1. In `LandmarkExtractor.kt`, replace `handLandmarker` with `leftHandLandmarker` and `rightHandLandmarker`.
2. Configure both of them exactly the same, but set `.setNumHands(1)` and point them to their respective callbacks (`onLeftHandResult` and `onRightHandResult`).

### Phase 3: The Cropping Pipeline (`extractAsync`)

When a new camera frame arrives, you will use the cached pose data to crop the bitmap before sending it to the hand landmarkers.

1. **Check for Cached Data:** If `lastLeftWristNormX == -1f` (e.g., the very first frame), you cannot crop. Send the full bitmap to both landmarkers.
2. **Calculate the Crop Boxes:** If you have cached data, convert the normalized wrist coordinates `[0.0 to 1.0]` into actual pixel coordinates by multiplying them by the `bitmap.width` and `bitmap.height`.
3. **Clamp the Boundaries:** The dynamic crop size (derived from shoulder width) might push the bounding box outside the image edge. You must write a clamping function:
```kotlin
val halfSize = lastDynamicCropSize / 2
var startX = (wristPixelX - halfSize).toInt()
var startY = (wristPixelY - halfSize).toInt()

// Clamp to boundaries to prevent crash
startX = maxOf(0, minOf(startX, bitmap.width - lastDynamicCropSize))
startY = maxOf(0, minOf(startY, bitmap.height - lastDynamicCropSize))

```


4. **Save to `FrameResult`:** Store `startX`, `startY`, and `lastDynamicCropSize` into the `FrameResult` object for this timestamp.
5. **Crop and Execute:** Use `Bitmap.createBitmap(bitmap, startX, startY, size, size)` to create the two tiny images, convert them to `MPImage`, and send them to `leftHandLandmarker` and `rightHandLandmarker` respectively.

### Phase 4: Cache the New Pose Data (`onPoseResult`)

Whenever the Pose landmarker finishes processing a frame, update the tracking variables for the *next* frame.

1. Inside `onPoseResult`, extract the raw coordinates for Landmark 15 (Left Wrist), Landmark 16 (Right Wrist), Landmark 11 (Left Shoulder), and Landmark 12 (Right Shoulder).
2. **Calculate Dynamic Size:** Calculate the pixel distance between the two shoulders using the Pythagorean theorem (`sqrt(dx^2 + dy^2)`). Multiply this distance by `1.5f` to create a comfortable tracking box size. Save this to `lastDynamicCropSize`.
3. **Cache the Wrists:** Save the normalized `x` and `y` coordinates of Landmarks 15 and 16 to your `lastLeftWristNormX`, etc.
* *Note: Only cache them if their `.visibility()` is > 0.5 to prevent tracking ghost limbs.*



### Phase 5: Coordinate Remapping (`onHandResult`)

When the hand callbacks fire, the coordinates they return will be relative to the tiny 160x160 crop. You must remap them back to the full 640x480 frame.

1. Inside your new `onLeftHandResult` and `onRightHandResult` callbacks, retrieve the `FrameResult` using the timestamp.
2. Loop through the 21 hand landmarks. Apply this exact math to translate them back to global space:
```kotlin
val rawCropX = result.landmarks()[0][j].x() // [0.0 to 1.0] relative to crop
val rawCropY = result.landmarks()[0][j].y() // [0.0 to 1.0] relative to crop

// 1. Convert to absolute pixels within the full image
val absolutePixelX = frame.leftCropStartX + (rawCropX * frame.leftCropSize)
val absolutePixelY = frame.leftCropStartY + (rawCropY * frame.leftCropSize)

// 2. Normalize back to full frame [0.0 to 1.0]
val fullFrameNormX = absolutePixelX / FULL_IMAGE_WIDTH.toFloat()
val fullFrameNormY = absolutePixelY / FULL_IMAGE_HEIGHT.toFloat()

```


3. Save `fullFrameNormX` and `fullFrameNormY` into the correct index of your `FloatArray(258)`.
4. Mark `frame.leftHandDone = true` and call `checkCompletion()`.

### Summary of Performance Gains

By implementing this exact flow, your CPU cores will process two ~200x200 images instead of two 640x480 images for hand tracking. This reduces the pixel processing load by roughly 80%. Because the coordinate remapping mathematically restores the landmarks to their global positions, `SignPredictor.kt` and your AI model will function perfectly without knowing the image was ever cropped.
