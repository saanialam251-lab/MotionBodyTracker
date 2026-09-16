package com.motiontracker.vision

import android.graphics.Bitmap
import android.graphics.Matrix
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch

data class LandmarkFrame(
    val hands: List<List<NormalizedLandmark>>,
    val pose: List<NormalizedLandmark>?,
    val motionPercent: Int,
    val motionBox: MotionBox?,
    val gesture: HandGesture = HandGesture.NONE,
    val mirrored: Boolean = true,
    // Dimensions of the coordinate space the landmarks / motionBox above are
    // normalized against: the camera frame *after* device-rotation
    // correction, matching what was actually fed to MediaPipe. 0 until the
    // first frame has been processed.
    val frameWidth: Int = 0,
    val frameHeight: Int = 0
)

class FrameAnalyzer(
    private val helper: LandmarkerHelper,
    private val scope: CoroutineScope,
    private val onUpdate: (motionPct: Int, hot: Boolean, label: String, gesture: HandGesture, actionGesture: HandGesture) -> Unit
) : ImageAnalysis.Analyzer {

    @Volatile var running = false
    @Volatile var handsOn = true
    @Volatile var bodyOn = true
    @Volatile var motionOn = true
    @Volatile var gesturesOn = true
    @Volatile var sensitivity = 12
    @Volatile var frontCamera = true

    @Volatile var lastHands = 0
    @Volatile var lastBody = false
    @Volatile var lastGesture = HandGesture.NONE
    @Volatile var lastFrame: LandmarkFrame? = null

    private val motion = MotionDetector()
    private var lastAlertAt = 0L
    private var lastGestureAt = 0L
    private var heldGesture = HandGesture.NONE
    private var heldSince = 0L

    // Tracks which stable gesture has already fired its one-shot action, so
    // holding a gesture (e.g. Peace) triggers the action once instead of on
    // every analyzed frame. Re-arms once the hand releases the gesture.
    private var firedGesture = HandGesture.NONE

    override fun analyze(image: ImageProxy) {
        try {
            val trackFull = running
            val rotationDegrees = image.imageInfo.rotationDegrees

            // "Display space" = the frame's dimensions after correcting for
            // device/sensor rotation — the same space the rotated bitmap fed
            // to MediaPipe uses, so landmarks and the motion box line up.
            val dispW: Int
            val dispH: Int
            if (rotationDegrees == 90 || rotationDegrees == 270) {
                dispW = image.height
                dispH = image.width
            } else {
                dispW = image.width
                dispH = image.height
            }

            var pct = 0
            var box: MotionBox? = null
            if (trackFull && motionOn) {
                val yPlane = image.planes[0]
                val res = motion.process(
                    yPlane.buffer, yPlane.rowStride, yPlane.pixelStride,
                    image.width, image.height
                )
                pct = res.percent
                box = res.box?.let { rotateToDisplay(it, rotationDegrees) }
            }

            val needBitmap = gesturesOn || (trackFull && (handsOn || bodyOn))
            val bitmap = if (needBitmap) image.toRotatedBitmap() else null
            if (bitmap != null) {
                if (gesturesOn || (trackFull && handsOn)) helper.detectHands(bitmap)
                if (trackFull && bodyOn) helper.detectPose(bitmap)
            }

            val hands = helper.latestHands?.landmarks() ?: emptyList()
            val pose = helper.latestPose?.landmarks()?.firstOrNull()
            lastHands = hands.size
            lastBody = pose != null

            val rawGesture = if (gesturesOn) GestureRecognizer.classifyFirst(hands) else HandGesture.NONE
            val stable = stabilize(rawGesture)
            lastGesture = stable
            val actionGesture = actionEdge(stable)

            lastFrame = LandmarkFrame(hands, pose, pct, box, stable, frontCamera, dispW, dispH)

            val hot = motionOn && pct >= sensitivity
            val now = System.currentTimeMillis()
            val label = when {
                stable != HandGesture.NONE && now - lastGestureAt > 1200 -> {
                    lastGestureAt = now
                    "Gesture: ${stable.name.replace('_', ' ')}"
                }
                hot && now - lastAlertAt > 2500 -> {
                    lastAlertAt = now
                    val where = if (hands.isNotEmpty() || pose != null) "on body part" else "in scene"
                    "Motion $where — $pct%"
                }
                else -> ""
            }
            scope.launch(Dispatchers.Main) { onUpdate(pct, hot, label, stable, actionGesture) }
        } catch (_: Exception) {
            // drop this frame
        } finally {
            image.close()
        }
    }

    private fun stabilize(current: HandGesture): HandGesture {
        val now = System.currentTimeMillis()
        if (current != heldGesture) {
            heldGesture = current
            heldSince = now
            return HandGesture.NONE
        }
        if (current == HandGesture.NONE) return HandGesture.NONE
        return if (now - heldSince >= 450) current else HandGesture.NONE
    }

    /**
     * Turns a continuously-held stable gesture into a one-shot edge trigger:
     * fires exactly once when [stable] first becomes non-NONE, then stays
     * silent (returns NONE) every subsequent frame the same gesture is held.
     * Re-arms once the hand releases the gesture and [stable] returns to NONE.
     */
    private fun actionEdge(stable: HandGesture): HandGesture {
        if (stable == HandGesture.NONE) {
            firedGesture = HandGesture.NONE
            return HandGesture.NONE
        }
        if (stable == firedGesture) return HandGesture.NONE
        firedGesture = stable
        return stable
    }

    /**
     * MotionDetector reads the raw, un-rotated sensor buffer directly (no
     * Bitmap conversion, for performance), so [box] is normalized against
     * the sensor's own width/height. Hand/pose landmarks, by contrast, come
     * from a bitmap we already rotated to match on-screen orientation.
     * Without this step the motion box would render sideways and in the
     * wrong place relative to the skeleton whenever [rotationDegrees] is 90
     * or 270 — the common case for a portrait-locked activity with a
     * landscape-mounted camera sensor (i.e. most phones).
     */
    private fun rotateToDisplay(box: MotionBox, rotationDegrees: Int): MotionBox {
        return when (rotationDegrees) {
            90 -> MotionBox(
                x = 1f - box.y - box.h,
                y = box.x,
                w = box.h,
                h = box.w
            )
            180 -> MotionBox(
                x = 1f - box.x - box.w,
                y = 1f - box.y - box.h,
                w = box.w,
                h = box.h
            )
            270 -> MotionBox(
                x = box.y,
                y = 1f - box.x - box.w,
                w = box.h,
                h = box.w
            )
            else -> box
        }
    }

    fun resetMotion() = motion.reset()
}

private fun ImageProxy.toRotatedBitmap(): Bitmap {
    val bmp = toBitmap()
    val degrees = imageInfo.rotationDegrees
    if (degrees == 0) return bmp
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
}
