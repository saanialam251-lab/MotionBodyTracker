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
import kotlin.math.sin

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

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        val f = frame ?: return
        val t = (System.currentTimeMillis() - startTime) / 1000f
        val pulse = (sin(t * 4f) + 1f) / 2f

        f.hands.forEach { lm ->
            LandmarkConnections.HAND.forEach { (a, b) ->
                if (a < lm.size && b < lm.size) {
                    drawBone(canvas, lm[a], lm[b], f.mirrored)
                }
            }
            lm.forEach { p ->
                canvas.drawCircle(mapX(p.x(), f.mirrored), p.y() * height, 9f, jointPaint)
            }
        }

        f.pose?.let { lm ->
            LandmarkConnections.POSE.forEach { (a, b) ->
                if (a < lm.size && b < lm.size && visible(lm[a]) && visible(lm[b])) {
                    drawBone(canvas, lm[a], lm[b], f.mirrored)
                }
            }
            lm.forEach { p ->
                if (visible(p)) {
                    canvas.drawCircle(mapX(p.x(), f.mirrored), p.y() * height, 9f, jointPaint)
                }
            }
        }

        f.motionBox?.let { b ->
            val left = mapX(b.x, f.mirrored)
            val right = mapX(b.x + b.w, f.mirrored)
            val l = minOf(left, right)
            val r = maxOf(left, right)
            val top = b.y * height
            val bottom = top + b.h * height
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

    private fun drawBone(canvas: Canvas, a: NormalizedLandmark, b: NormalizedLandmark, mirrored: Boolean) {
        canvas.drawLine(
            mapX(a.x(), mirrored), a.y() * height,
            mapX(b.x(), mirrored), b.y() * height,
            bonePaint
        )
    }

    private fun mapX(x: Float, mirrored: Boolean): Float {
        return if (mirrored) (1f - x) * width else x * width
    }

    private fun visible(p: NormalizedLandmark): Boolean {
        return try {
            p.visibility().orElse(1f) > 0.4f
        } catch (_: Exception) {
            true
        }
    }
}
