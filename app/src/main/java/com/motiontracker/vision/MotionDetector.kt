package com.motiontracker.vision

import kotlin.math.abs

data class MotionResult(
    val percent: Int,
    val box: MotionBox?
)

data class MotionBox(val x: Float, val y: Float, val w: Float, val h: Float)

/**
 * Frame differencing on the camera Y (luma) plane.
 * A pixel counts as moved when its brightness change exceeds [pixelThreshold].
 */
class MotionDetector(
    private val width: Int = 160,
    private val height: Int = 90,
    private val pixelThreshold: Int = 18
) {

    private var prev: ByteArray? = null

    fun reset() { prev = null }

    fun process(
        yBuffer: java.nio.ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        imgW: Int,
        imgH: Int
    ): MotionResult {
        val down = downscale(yBuffer, rowStride, pixelStride, imgW, imgH)
        val old = prev
        if (old == null) {
            prev = down
            return MotionResult(0, null)
        }

        var changed = 0
        var minX = width
        var minY = height
        var maxX = 0
        var maxY = 0
        val total = width * height
        for (i in 0 until total) {
            if (abs((down[i].toInt() and 0xFF) - (old[i].toInt() and 0xFF)) > pixelThreshold) {
                changed++
                val x = i % width
                val y = i / width
                if (x < minX) minX = x
                if (x > maxX) maxX = x
                if (y < minY) minY = y
                if (y > maxY) maxY = y
            }
        }
        prev = down
        val pct = (changed * 100) / total
        val box = if (pct > 0) MotionBox(
            minX.toFloat() / width,
            minY.toFloat() / height,
            (maxX - minX + 1).toFloat() / width,
            (maxY - minY + 1).toFloat() / height
        ) else null
        return MotionResult(pct, box)
    }

    // pixelStride matters because the Y plane's bytes aren't guaranteed to be
    // tightly packed (stride 1) on every device — some report a larger
    // pixelStride, in which case reading consecutive bytes without skipping
    // by it silently samples the wrong pixels.
    private fun downscale(
        y: java.nio.ByteBuffer,
        rowStride: Int,
        pixelStride: Int,
        imgW: Int,
        imgH: Int
    ): ByteArray {
        val out = ByteArray(width * height)
        val saved = y.position()
        val stride = if (pixelStride <= 0) 1 else pixelStride
        for (oy in 0 until height) {
            val sy = oy * imgH / height
            val rowBase = sy * rowStride
            for (ox in 0 until width) {
                val sx = ox * imgW / width
                val idx = rowBase + sx * stride
                if (idx in 0 until y.limit()) {
                    out[oy * width + ox] = y.get(idx)
                }
            }
        }
        y.position(saved)
        return out
    }
}
