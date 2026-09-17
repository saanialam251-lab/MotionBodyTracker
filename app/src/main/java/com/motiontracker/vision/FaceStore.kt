package com.motiontracker.vision

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.io.File

/**
 * On-device store for known face embeddings (name -> normalized embedding
 * vector), persisted as plain JSON in the app's private files dir. Only the
 * numeric embedding is stored — no images, nothing leaves the device.
 */
class FaceStore(context: Context) {

    private val file = File(context.filesDir, "known_faces.json")
    private val known = mutableMapOf<String, FloatArray>()

    init { load() }

    @Synchronized
    fun enroll(name: String, embedding: FloatArray) {
        if (name.isBlank()) return
        known[name.trim()] = embedding
        save()
    }

    @Synchronized
    fun remove(name: String) {
        known.remove(name)
        save()
    }

    @Synchronized
    fun names(): List<String> = known.keys.sorted()

    /** Best match at or above [threshold] cosine similarity, or null (i.e. "Unknown"). */
    @Synchronized
    fun match(embedding: FloatArray, threshold: Float = 0.62f): Pair<String, Float>? {
        var bestName: String? = null
        var bestScore = -1f
        for ((name, ref) in known) {
            val score = FaceHelper.cosineSimilarity(embedding, ref)
            if (score > bestScore) {
                bestScore = score
                bestName = name
            }
        }
        return if (bestName != null && bestScore >= threshold) bestName to bestScore else null
    }

    private fun load() {
        known.clear()
        if (!file.exists()) return
        try {
            val obj = JSONObject(file.readText())
            val keys = obj.keys()
            while (keys.hasNext()) {
                val name = keys.next()
                val arr = obj.getJSONArray(name)
                known[name] = FloatArray(arr.length()) { arr.getDouble(it).toFloat() }
            }
        } catch (_: Exception) {
            // corrupt or missing file — start fresh rather than crash
        }
    }

    private fun save() {
        try {
            val obj = JSONObject()
            for ((name, vec) in known) {
                val arr = JSONArray()
                for (v in vec) arr.put(v.toDouble())
                obj.put(name, arr)
            }
            file.writeText(obj.toString())
        } catch (_: Exception) {
            // best-effort persistence; not fatal if a write fails
        }
    }
}
