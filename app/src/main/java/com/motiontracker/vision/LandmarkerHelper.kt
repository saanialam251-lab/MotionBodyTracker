package com.motiontracker.vision

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarker
import com.google.mediapipe.tasks.vision.handlandmarker.HandLandmarkerResult
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarker
import com.google.mediapipe.tasks.vision.poselandmarker.PoseLandmarkerResult

/**
 * Wraps MediaPipe HandLandmarker + PoseLandmarker in LIVE_STREAM mode.
 * Model files must live in app/src/main/assets:
 *   hand_landmarker.task, pose_landmarker.task
 */
class LandmarkerHelper(context: Context) {

    @Volatile var latestHands: HandLandmarkerResult? = null
    @Volatile var latestPose: PoseLandmarkerResult? = null

    private val handLandmarker: HandLandmarker = HandLandmarker.createFromOptions(
        context,
        HandLandmarker.HandLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("hand_landmarker.task").build())
            .setNumHands(2)
            .setMinHandDetectionConfidence(0.5f)
            .setMinHandPresenceConfidence(0.5f)
            .setMinTrackingConfidence(0.5f)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { r, _ -> latestHands = r }
            .setErrorListener { e -> Log.e(TAG, e.message ?: "hand error") }
            .build()
    )

    private val poseLandmarker: PoseLandmarker = PoseLandmarker.createFromOptions(
        context,
        PoseLandmarker.PoseLandmarkerOptions.builder()
            .setBaseOptions(BaseOptions.builder().setModelAssetPath("pose_landmarker.task").build())
            .setNumPoses(1)
            // Raised from 0.5: at 0.5 the pose model was confident enough
            // to draw a full body skeleton over hanging bags/straps in a
            // cluttered shop scene — their silhouette loosely resembles a
            // shoulder-and-arm shape. Raising the bar cuts down on that kind
            // of false-positive "skeleton on an object" detection. Trade-off:
            // a real person in a similarly cluttered, low-contrast, or
            // partial-body shot may occasionally need to be a bit more
            // clearly posed/visible before Body switches to "yes".
            .setMinPoseDetectionConfidence(0.65f)
            .setMinPosePresenceConfidence(0.65f)
            .setMinTrackingConfidence(0.5f)
            .setRunningMode(RunningMode.LIVE_STREAM)
            .setResultListener { r, _ -> latestPose = r }
            .setErrorListener { e -> Log.e(TAG, e.message ?: "pose error") }
            .build()
    )

    fun detectHands(bitmap: Bitmap) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        handLandmarker.detectAsync(mpImage, SystemClock.uptimeMillis())
    }

    fun detectPose(bitmap: Bitmap) {
        val mpImage = BitmapImageBuilder(bitmap).build()
        poseLandmarker.detectAsync(mpImage, SystemClock.uptimeMillis())
    }

    fun close() {
        try { handLandmarker.close() } catch (_: Exception) {}
        try { poseLandmarker.close() } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "Landmarker"

        fun modelsExist(context: Context): Boolean {
            return try {
                context.assets.open("hand_landmarker.task").close()
                context.assets.open("pose_landmarker.task").close()
                true
            } catch (_: Exception) {
                false
            }
        }
    }
}
