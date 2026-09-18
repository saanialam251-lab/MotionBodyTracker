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
/**
 * On-device store for known face embeddings (name -> a handful of normalized
 * embedding vectors), persisted as plain JSON in the app's private files
 * dir. Only the numeric embeddings are stored — no images, nothing leaves
 * the device.
 *
 * Each name keeps up to [MAX_SAMPLES] embeddings rather than a single one.
 * A single enrollment sample is one lighting/angle/expression snapshot of a
 * face; matching new frames against just that one sample means anything
 * that drifts from it (a slightly different angle, a shadow) can drop below
 * threshold and read as "Unknown" even though it's the same person. Enrolling
 * the same name again — from a different angle — adds another sample rather
 * than overwriting the first, and matching takes the *best* similarity
 * across all of a name's samples, which is far less sensitive to any one
 * sample being a bad fit for the current frame.
 */
class FaceStore(context: Context) {

    private val file = File(context.filesDir, "known_faces.json")
    private val known = mutableMapOf<String, MutableList<FloatArray>>()

    init { load() }

    @Synchronized
    fun enroll(name: String, embedding: FloatArray) {
        val trimmed = name.trim()
        if (trimmed.isBlank()) return
        val samples = known.getOrPut(trimmed) { mutableListOf() }
        // Skip near-duplicate samples (e.g. two enrolls in a row with barely
        // any head movement) so the cap below isn't wasted on redundant data.
        if (samples.none { FaceHelper.cosineSimilarity(it, embedding) > 0.97f }) {
            samples.add(embedding)
            while (samples.size > MAX_SAMPLES) samples.removeAt(0)
        }
        save()
    }

    @Synchronized
    fun remove(name: String) {
        known.remove(name)
        save()
    }

    @Synchronized
    fun names(): List<String> = known.keys.sorted()

    /** Best match at or above [threshold] cosine similarity, or null (i.e. "Unknown").
     * Compares against every stored sample of every name and keeps the single
     * best score, so one good match from any angle is enough. */
    @Synchronized
    fun match(embedding: FloatArray, threshold: Float = 0.62f): Pair<String, Float>? {
        var bestName: String? = null
        var bestScore = -1f
        for ((name, samples) in known) {
            for (ref in samples) {
                val score = FaceHelper.cosineSimilarity(embedding, ref)
                if (score > bestScore) {
                    bestScore = score
                    bestName = name
                }
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
                val value = obj.get(name)
                // Back-compat: old format stored one flat JSONArray of floats
                // per name; new format stores a JSONArray of those (one per
                // sample). Detect which we're looking at from the first element.
                val samples = mutableListOf<FloatArray>()
                if (value is JSONArray) {
                    val looksLikeSampleList = value.length() > 0 && value.opt(0) is JSONArray
                    if (looksLikeSampleList) {
                        for (i in 0 until value.length()) {
                            val arr = value.getJSONArray(i)
                            samples.add(FloatArray(arr.length()) { arr.getDouble(it).toFloat() })
                        }
                    } else {
                        samples.add(FloatArray(value.length()) { value.getDouble(it).toFloat() })
                    }
                }
                if (samples.isNotEmpty()) known[name] = samples
            }
        } catch (_: Exception) {
            // corrupt or missing file — start fresh rather than crash
        }
    }

    private fun save() {
        try {
            val obj = JSONObject()
            for ((name, samples) in known) {
                val outer = JSONArray()
                for (vec in samples) {
                    val arr = JSONArray()
                    for (v in vec) arr.put(v.toDouble())
                    outer.put(arr)
                }
                obj.put(name, outer)
            }
            file.writeText(obj.toString())
        } catch (_: Exception) {
            // best-effort persistence; not fatal if a write fails
        }
    }

    companion object {
        private const val MAX_SAMPLES = 5
    }
}
