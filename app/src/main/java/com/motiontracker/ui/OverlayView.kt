package com.motiontracker.ui

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.DashPathEffect
import android.graphics.Paint
import android.graphics.Typeface
import android.util.AttributeSet
import android.view.View
import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import com.motiontracker.vision.HandGesture
import com.motiontracker.vision.LandmarkConnections
import com.motiontracker.vision.LandmarkFrame
import kotlin.math.max
import kotlin.math.sin

/**
 * Green-skeleton / red-joint overlay plus motion box with a pulsing glow.
 *
 * The camera preview (PreviewView) uses ScaleType.FILL_CENTER, which scales
 * the camera image to fill the view and crops the overflow rather than
 * stretching it to fit. To stay lined up with what's actually on screen,
 * every normalized landmark/box coordinate here goes through the same
 * "scale to cover, then center" transform instead of being stretched
 * directly across the view's full width/height.
 */
class OverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null
) : View(context, attrs) {

    @Volatile var frame: LandmarkFrame? = null

    private val bonePaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#00E676")
        strokeWidth = 5f
        style = Paint.Style.STROKE
        strokeCap = Paint.Cap.ROUND
    }
    private val jointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.parseColor("#FF4444")
        style = Paint.Style.FILL
    }
    private val glowPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
    }
    private val boxPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        style = Paint.Style.STROKE
        strokeWidth = 6f
        pathEffect = DashPathEffect(floatArrayOf(18f, 12f), 0f)
    }
    private val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.WHITE
        textSize = 36f
        typeface = Typeface.DEFAULT_BOLD
    }

    private var startTime = System.currentTimeMillis()

    /** Scale-to-cover transform matching PreviewView's FILL_CENTER, plus the
     *  source dimensions it was computed against. */
    private class Transform(val scale: Float, val offsetX: Float, val offsetY: Float, val srcW: Float, val srcH: Float)

    private fun transformFor(f: LandmarkFrame): Transform {
        val vw = width.toFloat().coerceAtLeast(1f)
        val vh = height.toFloat().coerceAtLeast(1f)
        // Before the first frame's real dimensions are known, fall back to
        // treating the view itself as the source (scale 1, no offset) —
        // i.e. the old, naive stretch-to-fit behavior — rather than divide
        // by zero.
        val srcW = if (f.frameWidth > 0) f.frameWidth.toFloat() else vw
        val srcH = if (f.frameHeight > 0) f.frameHeight.toFloat() else vh
        val scale = max(vw / srcW, vh / srcH)
        val offsetX = (vw - srcW * scale) / 2f
        val offsetY = (vh - srcH * scale) / 2f
        return Transform(scale, offsetX, offsetY, srcW, srcH)
    }

    private fun mapX(nx: Float, t: Transform, mirrored: Boolean): Float {
        val px = nx * t.srcW * t.scale + t.offsetX
        return if (mirrored) width - px else px
    }

    private fun mapY(ny: Float, t: Transform): Float {
        return ny * t.srcH * t.scale + t.offsetY
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val f = frame ?: return
        val t = transformFor(f)
        val time = (System.currentTimeMillis() - startTime) / 1000f
        val pulse = (sin(time * 4f) + 1f) / 2f

        f.hands.forEach { lm ->
            LandmarkConnections.HAND.forEach { (a, b) ->
                if (a < lm.size && b < lm.size) {
                    drawBone(canvas, lm[a], lm[b], t, f.mirrored)
                }
            }
            lm.forEach { p ->
                canvas.drawCircle(mapX(p.x(), t, f.mirrored), mapY(p.y(), t), 9f, jointPaint)
            }
        }

        f.pose?.let { lm ->
            LandmarkConnections.POSE.forEach { (a, b) ->
                if (a < lm.size && b < lm.size && visible(lm[a]) && visible(lm[b])) {
                    drawBone(canvas, lm[a], lm[b], t, f.mirrored)
                }
            }
            lm.forEach { p ->
                if (visible(p)) {
                    canvas.drawCircle(mapX(p.x(), t, f.mirrored), mapY(p.y(), t), 9f, jointPaint)
                }
            }
        }

        f.motionBox?.let { b ->
            val left = mapX(b.x, t, f.mirrored)
            val right = mapX(b.x + b.w, t, f.mirrored)
            val l = minOf(left, right)
            val r = maxOf(left, right)
            val top = mapY(b.y, t)
            val bottom = mapY(b.y + b.h, t)
            val hot = f.motionPercent >= 8
            val col = if (hot) Color.parseColor("#F85149") else Color.parseColor("#3FB950")
            glowPaint.color = col
            glowPaint.alpha = (60 + pulse * 120).toInt()
            glowPaint.strokeWidth = 14f + pulse * 12f
            canvas.drawRect(l, top, r, bottom, glowPaint)
            boxPaint.color = col
            canvas.drawRect(l, top, r, bottom, boxPaint)
            if (hot) {
                canvas.drawText(
                    "MOTION ${f.motionPercent}%",
                    l,
                    (top - 14f).coerceAtLeast(50f),
                    textPaint
                )
            }
        }

        if (f.gesture != HandGesture.NONE) {
            canvas.drawText(
                f.gesture.name.replace('_', ' '),
                24f,
                height - 36f,
                textPaint
            )
        }
    }

    fun snapshot(): Bitmap {
        val bmp = Bitmap.createBitmap(width.coerceAtLeast(1), height.coerceAtLeast(1), Bitmap.Config.ARGB_8888)
        draw(Canvas(bmp))
        return bmp
    }

    private fun drawBone(canvas: Canvas, a: NormalizedLandmark, b: NormalizedLandmark, t: Transform, mirrored: Boolean) {
        canvas.drawLine(
            mapX(a.x(), t, mirrored), mapY(a.y(), t),
            mapX(b.x(), t, mirrored), mapY(b.y(), t),
            bonePaint
        )
    }

    private fun visible(p: NormalizedLandmark): Boolean {
        return try {
            p.visibility().orElse(1f) > 0.4f
        } catch (_: Exception) {
            true
        }
    }
}
