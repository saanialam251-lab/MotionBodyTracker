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
    val mirrored: Boolean = true
)

class FrameAnalyzer(
    private val helper: LandmarkerHelper,
    private val scope: CoroutineScope,
    private val onUpdate: (motionPct: Int, hot: Boolean, label: String, gesture: HandGesture) -> Unit
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

    override fun analyze(image: ImageProxy) {
        try {
            val trackFull = running
            var pct = 0
            var box: MotionBox? = null
            if (trackFull && motionOn) {
                val yPlane = image.planes[0]
                val res = motion.process(yPlane.buffer, yPlane.rowStride, image.width, image.height)
                pct = res.percent
                box = res.box
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

            lastFrame = LandmarkFrame(hands, pose, pct, box, stable, frontCamera)

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
            scope.launch(Dispatchers.Main) { onUpdate(pct, hot, label, stable) }
        } catch (_: Exception) {
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

    fun resetMotion() = motion.reset()
}

private fun ImageProxy.toRotatedBitmap(): Bitmap {
    val bmp = toBitmap()
    val degrees = imageInfo.rotationDegrees
    if (degrees == 0) return bmp
    val matrix = Matrix().apply { postRotate(degrees.toFloat()) }
    return Bitmap.createBitmap(bmp, 0, 0, bmp.width, bmp.height, matrix, true)
}
