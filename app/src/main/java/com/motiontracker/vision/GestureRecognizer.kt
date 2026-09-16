package com.motiontracker.vision

import com.google.mediapipe.tasks.components.containers.NormalizedLandmark
import kotlin.math.hypot

enum class HandGesture {
    NONE,
    OPEN_PALM,
    PEACE,
    FIST
}

object GestureRecognizer {

    private const val WRIST = 0
    private const val THUMB_TIP = 4
    private const val THUMB_IP = 3
    private const val INDEX_TIP = 8
    private const val INDEX_PIP = 6
    private const val MIDDLE_TIP = 12
    private const val MIDDLE_PIP = 10
    private const val RING_TIP = 16
    private const val RING_PIP = 14
    private const val PINKY_TIP = 20
    private const val PINKY_PIP = 18

    fun classify(hand: List<NormalizedLandmark>): HandGesture {
        if (hand.size < 21) return HandGesture.NONE
        val wrist = hand[WRIST]
        fun extended(tip: Int, pip: Int): Boolean {
            return dist(hand[tip], wrist) > dist(hand[pip], wrist) * 1.12f
        }

        val thumb = dist(hand[THUMB_TIP], wrist) > dist(hand[THUMB_IP], wrist) * 1.05f
        val index = extended(INDEX_TIP, INDEX_PIP)
        val middle = extended(MIDDLE_TIP, MIDDLE_PIP)
        val ring = extended(RING_TIP, RING_PIP)
        val pinky = extended(PINKY_TIP, PINKY_PIP)

        val raised = listOf(index, middle, ring, pinky).count { it }

        return when {
            raised >= 4 && thumb -> HandGesture.OPEN_PALM
            index && middle && !ring && !pinky -> HandGesture.PEACE
            raised == 0 && !thumb -> HandGesture.FIST
            else -> HandGesture.NONE
        }
    }

    fun classifyFirst(hands: List<List<NormalizedLandmark>>): HandGesture {
        for (hand in hands) {
            val g = classify(hand)
            if (g != HandGesture.NONE) return g
        }
        return HandGesture.NONE
    }

    private fun dist(a: NormalizedLandmark, b: NormalizedLandmark): Float {
        val dx = a.x() - b.x()
        val dy = a.y() - b.y()
        return hypot(dx, dy)
    }
}
