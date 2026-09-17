package com.motiontracker.vision

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.core.RunningMode
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import kotlin.math.sqrt

/** One detected face, with a normalized ([0,1]) bounding box in the same
 * "display space" as hand/pose landmarks and the motion box. [name] is null
 * when there's no embedder loaded (detector-only mode), "Unknown" when the
 * embedder found no match above threshold, or the enrolled person's name.
 * [embedding] is kept around so the UI can enroll the face just seen. */
data class FaceMatch(
    val box: android.graphics.RectF,
    val name: String?,
    val confidence: Float,
    val embedding: FloatArray? = null
)

/**
 * Wraps MediaPipe FaceDetector (bounding boxes) plus an optional bundled
 * TFLite face-embedding model for known/unknown recognition.
 *
 * Model files expected in app/src/main/assets:
 *   face_detection_short_range.tflite   required — enables face boxes at all
 *   face_recognition.tflite             optional — enables known/unknown
 *                                        matching; without it every detected
 *                                        face is reported as "Unknown" and
 *                                        can't be enrolled by name.
 *
 * Both are absent by default (like the hand/pose models) — see
 * PUT_MODELS_HERE.txt for download links. Either piece missing degrades
 * gracefully instead of crashing: no detector -> no faces at all; no
 * embedder -> boxes with no name/enroll capability.
 *
 * The embedder assumes a MobileFaceNet-style model: roughly 112x112 RGB
 * input normalized to [-1,1], one face in -> one float embedding vector out.
 * The exact input size and embedding length are read from the model file
 * itself at load time (see embedInputSize/embedOutputSize below), so models
 * with a different resolution or embedding length than the 112/192 defaults
 * work with no code changes — only the pixel normalization below is a fixed
 * assumption, since TFLite doesn't expose that in the model's shape info.
 */
class FaceHelper(context: Context) {

    @Volatile var latestFaces: FaceDetectorResult? = null

    private val faceDetector: FaceDetector? = try {
        if (detectorModelExists(context)) {
            FaceDetector.createFromOptions(
                context,
                FaceDetector.FaceDetectorOptions.builder()
                    .setBaseOptions(BaseOptions.builder().setModelAssetPath(FACE_DETECT_MODEL).build())
                    .setMinDetectionConfidence(0.5f)
                    .setRunningMode(RunningMode.LIVE_STREAM)
                    .setResultListener { r, _ -> latestFaces = r }
                    .setErrorListener { e -> Log.e(TAG, e.message ?: "face detector error") }
                    .build()
            )
        } else null
    } catch (e: Exception) {
        Log.e(TAG, "Face detector unavailable: ${e.message}")
        null
    }

    private val embedder: Interpreter? = try {
        loadEmbedderModel(context)?.let { Interpreter(it) }
    } catch (e: Exception) {
        Log.e(TAG, "Face embedder unavailable: ${e.message}")
        null
    }

    // Read the model's *actual* input/output tensor shapes at load time
    // instead of assuming fixed sizes or a fixed layout. Two things
    // previously broke silently: (1) hardcoding EMBED_OUTPUT_SIZE = 192
    // against models — like MobileFaceNet — that output 128 numbers, and
    // (2) assuming TFLite's usual channel-last [1,H,W,3] layout when a
    // model exported from PyTorch (channel-first natively) keeps a
    // [1,3,H,W] input instead. Both are now detected from the model file
    // itself, so mismatched shape/layout can't silently fail anymore.
    private data class InputLayout(val nchw: Boolean, val height: Int, val width: Int)

    private val inputLayout: InputLayout by lazy {
        val fallback = InputLayout(nchw = false, height = EMBED_INPUT_SIZE, width = EMBED_INPUT_SIZE)
        try {
            val shape = embedder?.getInputTensor(0)?.shape()
            when {
                shape == null || shape.size != 4 -> fallback
                shape[3] == 3 -> InputLayout(nchw = false, height = shape[1], width = shape[2])
                shape[1] == 3 -> InputLayout(nchw = true, height = shape[2], width = shape[3])
                else -> fallback
            }
        } catch (e: Exception) {
            Log.e(TAG, "Could not read embedder input shape, defaulting to ${EMBED_INPUT_SIZE}x$EMBED_INPUT_SIZE NHWC: ${e.message}")
            fallback
        }
    }
    private val embedOutputSize: Int by lazy {
        try {
            embedder?.getOutputTensor(0)?.shape()?.lastOrNull() ?: EMBED_OUTPUT_SIZE
        } catch (e: Exception) {
            Log.e(TAG, "Could not read embedder output shape, defaulting to $EMBED_OUTPUT_SIZE: ${e.message}")
            EMBED_OUTPUT_SIZE
        }
    }

    /** Set whenever [embed] fails, so callers (the Enroll button in
     * particular) can show the real reason instead of a generic message. */
    @Volatile var lastEmbedError: String? = null
        private set

    val hasDetector: Boolean get() = faceDetector != null
    val hasEmbedder: Boolean get() = embedder != null

    fun detect(bitmap: Bitmap) {
        val detector = faceDetector ?: return
        val mpImage = BitmapImageBuilder(bitmap).build()
        detector.detectAsync(mpImage, SystemClock.uptimeMillis())
    }

    /**
     * Crops [bitmap] to [pixelBox] and runs the embedder, returning an
     * L2-normalized embedding vector, or null if there's no embedder loaded
     * or the crop is degenerate.
     */
    fun embed(bitmap: Bitmap, pixelBox: Rect): FloatArray? {
        val net = embedder ?: return null
        val layout = inputLayout
        val w = layout.width
        val h = layout.height
        val left = pixelBox.left.coerceIn(0, bitmap.width - 1)
        val top = pixelBox.top.coerceIn(0, bitmap.height - 1)
        val right = pixelBox.right.coerceIn(left + 1, bitmap.width)
        val bottom = pixelBox.bottom.coerceIn(top + 1, bitmap.height)
        val crop = try {
            Bitmap.createBitmap(bitmap, left, top, right - left, bottom - top)
        } catch (e: Exception) {
            lastEmbedError = "Couldn't crop face region: ${e.message}"
            return null
        }
        val resized = Bitmap.createScaledBitmap(crop, w, h, true)
        val pixels = IntArray(w * h)
        resized.getPixels(pixels, 0, w, 0, 0, w, h)

        val input = ByteBuffer.allocateDirect(4 * w * h * 3).order(ByteOrder.nativeOrder())
        if (layout.nchw) {
            // Channel-first (PyTorch-style) layout: all R values, then all
            // G, then all B — as opposed to TFLite's usual interleaved
            // per-pixel RGB. Getting this backwards produces the same
            // "wrong total byte count" crash as a wrong H/W would.
            for (p in pixels) input.putFloat((((p shr 16) and 0xFF) - 127.5f) / 128f)
            for (p in pixels) input.putFloat((((p shr 8) and 0xFF) - 127.5f) / 128f)
            for (p in pixels) input.putFloat(((p and 0xFF) - 127.5f) / 128f)
        } else {
            for (p in pixels) {
                input.putFloat((((p shr 16) and 0xFF) - 127.5f) / 128f)
                input.putFloat((((p shr 8) and 0xFF) - 127.5f) / 128f)
                input.putFloat(((p and 0xFF) - 127.5f) / 128f)
            }
        }
        input.rewind()
        val output = Array(1) { FloatArray(embedOutputSize) }
        return try {
            net.run(input, output)
            lastEmbedError = null
            l2Normalize(output[0])
        } catch (e: Exception) {
            val msg = "input=${w}x$h (${if (layout.nchw) "NCHW" else "NHWC"}) output=$embedOutputSize: ${e.message}"
            Log.e(TAG, "Embedding failed: $msg")
            lastEmbedError = msg
            null
        }
    }

    fun close() {
        try { faceDetector?.close() } catch (_: Exception) {}
        try { embedder?.close() } catch (_: Exception) {}
    }

    companion object {
        private const val TAG = "FaceHelper"
        private const val FACE_DETECT_MODEL = "face_detection_short_range.tflite"
        private const val FACE_EMBED_MODEL = "face_recognition.tflite"
        const val EMBED_INPUT_SIZE = 112   // fallback if the model's shape can't be read
        const val EMBED_OUTPUT_SIZE = 192  // fallback if the model's shape can't be read

        fun detectorModelExists(context: Context): Boolean = try {
            context.assets.open(FACE_DETECT_MODEL).close(); true
        } catch (_: Exception) { false }

        fun embedderModelExists(context: Context): Boolean = try {
            context.assets.open(FACE_EMBED_MODEL).close(); true
        } catch (_: Exception) { false }

        private fun loadEmbedderModel(context: Context): MappedByteBuffer? {
            if (!embedderModelExists(context)) return null
            val afd = context.assets.openFd(FACE_EMBED_MODEL)
            return FileInputStream(afd.fileDescriptor).use { input ->
                input.channel.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            }
        }

        fun l2Normalize(v: FloatArray): FloatArray {
            var sum = 0f
            for (x in v) sum += x * x
            val norm = sqrt(sum).coerceAtLeast(1e-6f)
            return FloatArray(v.size) { v[it] / norm }
        }

        /** Both vectors are expected to already be L2-normalized, so their dot
         * product equals cosine similarity. */
        fun cosineSimilarity(a: FloatArray, b: FloatArray): Float {
            if (a.size != b.size) return -1f
            var dot = 0f
            for (i in a.indices) dot += a[i] * b[i]
            return dot
        }
    }
}
