package com.motiontracker

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.motiontracker.vision.FaceHelper
import com.motiontracker.vision.FaceStore
import com.motiontracker.vision.FrameAnalyzer
import com.motiontracker.vision.HandGesture
import com.motiontracker.vision.LandmarkFrame
import com.motiontracker.vision.LandmarkerHelper
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

data class TrackerUiState(
    val running: Boolean = false,
    val frontCamera: Boolean = true,
    val handsOn: Boolean = true,
    val bodyOn: Boolean = true,
    val motionOn: Boolean = true,
    val gesturesOn: Boolean = true,
    val faceOn: Boolean = true,
    val sensitivity: Int = 12,
    val handsFound: Int = 0,
    val bodyFound: Boolean = false,
    val motionPercent: Int = 0,
    val motionHot: Boolean = false,
    val gesture: HandGesture = HandGesture.NONE,
    val snapshotRequest: Long = 0L,
    val frame: LandmarkFrame? = null,
    val log: List<String> = emptyList(),
    val modelsReady: Boolean = true,
    // Face detector model present? If false, the Face toggle does nothing.
    val faceModelReady: Boolean = false,
    // Face recognition (embedding) model present? If false, faces are
    // detected (boxes shown) but never labeled known/"Unknown" and can't be
    // enrolled — see FaceHelper's doc comment for the required asset.
    val faceEmbedderReady: Boolean = false,
    val facesFound: Int = 0,
    val faceNames: List<String> = emptyList(),
    val knownFaceNames: List<String> = emptyList(),
    val enrollMessage: String? = null
)

class TrackerViewModel(app: Application) : AndroidViewModel(app) {

    // Kept as an explicit property (rather than relying on the bare
    // constructor parameter) so it's reachable from member functions like
    // enrollFace(), not just from code inside init{}.
    private val appContext: Application = app

    private val _ui = MutableStateFlow(TrackerUiState())
    val ui: StateFlow<TrackerUiState> = _ui.asStateFlow()

    lateinit var analyzer: FrameAnalyzer
        private set
    private lateinit var helper: LandmarkerHelper
    private lateinit var faceHelper: FaceHelper
    private lateinit var faceStore: FaceStore

    init {
        if (!LandmarkerHelper.modelsExist(app)) {
            _ui.value = TrackerUiState(
                modelsReady = false,
                log = listOf("Model files missing — see README")
            )
        } else {
            try {
                helper = LandmarkerHelper(app)
                faceHelper = FaceHelper(app)
                faceStore = FaceStore(app)
                val faceModelReady = FaceHelper.detectorModelExists(app)
                val faceEmbedderReady = FaceHelper.embedderModelExists(app)

                analyzer = FrameAnalyzer(helper, faceHelper, faceStore, viewModelScope) {
                    motionPct, hot, label, gesture, actionGesture, unknownFaceAlert, motionSnapshotTrigger ->

                    val current = _ui.value
                    var running = current.running
                    var snapshotRequest = current.snapshotRequest
                    if (current.gesturesOn) {
                        // actionGesture is edge-triggered by FrameAnalyzer: it is only
                        // non-NONE on the single frame a gesture first becomes stable,
                        // so each branch below fires once per palm/fist, not once
                        // per analyzed frame the gesture stays held.
                        when (actionGesture) {
                            HandGesture.OPEN_PALM -> if (!running) {
                                running = true
                                analyzer.running = true
                                analyzer.resetMotion()
                            }
                            HandGesture.FIST -> if (running) {
                                running = false
                                analyzer.running = false
                            }
                            else -> {}
                        }
                    }

                    // Snapshot now fires off motion itself (the motion box
                    // turning red / "hot") instead of requiring the Peace
                    // gesture — motionSnapshotTrigger is already edge/cooldown
                    // gated in FrameAnalyzer, so this fires once per motion
                    // event rather than every frame motion stays hot.
                    if (running && motionSnapshotTrigger) {
                        snapshotRequest = System.currentTimeMillis()
                    }

                    if (unknownFaceAlert) {
                        NotificationHelper.notifyUnknownFace(appContext)
                    }

                    var newLog = if (label.isNotBlank()) {
                        (listOf(label) + current.log).distinct().take(6)
                    } else current.log
                    if (unknownFaceAlert) {
                        newLog = (listOf("Unknown face detected") + newLog).distinct().take(6)
                    }

                    _ui.value = current.copy(
                        running = running,
                        snapshotRequest = snapshotRequest,
                        motionPercent = motionPct,
                        motionHot = hot,
                        handsFound = analyzer.lastHands,
                        bodyFound = analyzer.lastBody,
                        gesture = gesture,
                        frame = analyzer.lastFrame,
                        log = newLog,
                        facesFound = analyzer.lastFaces.size,
                        faceNames = analyzer.lastFaces.map { it.name ?: "Face" }
                    )
                }
                analyzer.faceOn = faceModelReady
                _ui.value = _ui.value.copy(
                    faceModelReady = faceModelReady,
                    faceEmbedderReady = faceEmbedderReady,
                    faceOn = faceModelReady,
                    knownFaceNames = faceStore.names()
                )
            } catch (e: Exception) {
                _ui.value = TrackerUiState(
                    modelsReady = false,
                    log = listOf("Failed to load models: ${e.message}")
                )
            }
        }
    }

    fun setRunning(r: Boolean) {
        _ui.value = _ui.value.copy(running = r)
        if (::analyzer.isInitialized) {
            analyzer.running = r
            if (r) analyzer.resetMotion()
        }
    }

    fun flipCamera() {
        val next = !_ui.value.frontCamera
        _ui.value = _ui.value.copy(frontCamera = next)
        if (::analyzer.isInitialized) analyzer.frontCamera = next
    }

    fun toggleHands() {
        val next = !_ui.value.handsOn
        _ui.value = _ui.value.copy(handsOn = next)
        if (::analyzer.isInitialized) analyzer.handsOn = next
    }

    fun toggleBody() {
        val next = !_ui.value.bodyOn
        _ui.value = _ui.value.copy(bodyOn = next)
        if (::analyzer.isInitialized) analyzer.bodyOn = next
    }

    fun toggleMotion() {
        val next = !_ui.value.motionOn
        _ui.value = _ui.value.copy(motionOn = next)
        if (::analyzer.isInitialized) analyzer.motionOn = next
    }

    fun toggleGestures() {
        val next = !_ui.value.gesturesOn
        _ui.value = _ui.value.copy(gesturesOn = next)
        if (::analyzer.isInitialized) analyzer.gesturesOn = next
    }

    fun toggleFace() {
        if (!_ui.value.faceModelReady) return
        val next = !_ui.value.faceOn
        _ui.value = _ui.value.copy(faceOn = next)
        if (::analyzer.isInitialized) analyzer.faceOn = next
    }

    fun setSensitivity(v: Int) {
        _ui.value = _ui.value.copy(sensitivity = v)
        if (::analyzer.isInitialized) analyzer.sensitivity = v
    }

    fun snapshotConsumed() {
        _ui.value = _ui.value.copy(snapshotRequest = 0L)
    }

    /** Enrolls [name] using the embedding of the most recently seen face.
     * No-op (with an explanatory [TrackerUiState.enrollMessage]) if there's
     * no face in frame or the recognition model isn't bundled. */
    fun enrollFace(name: String) {
        val trimmed = name.trim()
        if (trimmed.isEmpty()) return
        if (!::analyzer.isInitialized || !::faceStore.isInitialized || !::faceHelper.isInitialized) return
        val embedding = analyzer.lastFaces.firstOrNull { it.embedding != null }?.embedding
        if (embedding == null) {
            val reason = faceHelper.lastEmbedError
            _ui.value = _ui.value.copy(
                enrollMessage = when {
                    !_ui.value.faceEmbedderReady ->
                        "Face recognition model not installed — see README"
                    reason != null ->
                        "Recognition model error: $reason"
                    analyzer.lastFaces.isEmpty() ->
                        "No face detected right now — look at the camera and try again"
                    else ->
                        "Couldn't read that face — try again with better lighting"
                }
            )
            return
        }
        faceStore.enroll(trimmed, embedding)
        _ui.value = _ui.value.copy(
            enrollMessage = "Enrolled \"$trimmed\"",
            knownFaceNames = faceStore.names()
        )
    }

    fun clearEnrollMessage() {
        _ui.value = _ui.value.copy(enrollMessage = null)
    }

    override fun onCleared() {
        super.onCleared()
        if (::helper.isInitialized) helper.close()
        if (::faceHelper.isInitialized) faceHelper.close()
    }
}
