package com.motiontracker.vision

object LandmarkConnections {

    val HAND = listOf(
        0 to 1, 1 to 2, 2 to 3, 3 to 4,
        0 to 5, 5 to 6, 6 to 7, 7 to 8,
        5 to 9, 9 to 10, 10 to 11, 11 to 12,
        9 to 13, 13 to 14, 14 to 15, 15 to 16,
        13 to 17, 0 to 17, 17 to 18, 18 to 19, 19 to 20
    )

    val POSE = listOf(
        11 to 12,
        11 to 13, 13 to 15,
        12 to 14, 14 to 16,
        15 to 17, 15 to 19, 15 to 21,
        16 to 18, 16 to 20, 16 to 22,
        11 to 23, 12 to 24, 23 to 24,
        23 to 25, 25 to 27,
        24 to 26, 26 to 28,
        27 to 29, 27 to 31,
        28 to 30, 28 to 32,
        7 to 3, 3 to 2, 2 to 1, 1 to 0, 0 to 4, 4 to 5, 5 to 6, 6 to 8,
        9 to 10
    )
}
