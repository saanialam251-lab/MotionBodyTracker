package com.motiontracker

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
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
    val sensitivity: Int = 12,
    val handsFound: Int = 0,
    val bodyFound: Boolean = false,
    val motionPercent: Int = 0,
    val motionHot: Boolean = false,
    val gesture: HandGesture = HandGesture.NONE,
    val snapshotRequest: Long = 0L,
    val frame: LandmarkFrame? = null,
    val log: List<String> = emptyList(),
    val modelsReady: Boolean = true
)

class TrackerViewModel(app: Application) : AndroidViewModel(app) {

    private val _ui = MutableStateFlow(TrackerUiState())
    val ui: StateFlow<TrackerUiState> = _ui.asStateFlow()

    lateinit var analyzer: FrameAnalyzer
        private set
    private lateinit var helper: LandmarkerHelper

    init {
        if (!LandmarkerHelper.modelsExist(app)) {
            _ui.value = TrackerUiState(
                modelsReady = false,
                log = listOf("Model files missing — see README")
            )
        } else {
            try {
                helper = LandmarkerHelper(app)
                analyzer = FrameAnalyzer(helper, viewModelScope) { motionPct, hot, label, gesture, actionGesture ->
                    val current = _ui.value
                    var running = current.running
                    var snapshotRequest = current.snapshotRequest
                    if (current.gesturesOn) {
                        // actionGesture is edge-triggered by FrameAnalyzer: it is only
                        // non-NONE on the single frame a gesture first becomes stable,
                        // so each branch below fires once per palm/peace/fist, not once
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
                            HandGesture.PEACE -> if (running) snapshotRequest = System.currentTimeMillis()
                            else -> {}
                        }
                    }
                    val newLog = if (label.isNotBlank()) {
                        (listOf(label) + current.log).distinct().take(6)
                    } else current.log
                    _ui.value = current.copy(
                        running = running,
                        snapshotRequest = snapshotRequest,
                        motionPercent = motionPct,
                        motionHot = hot,
                        handsFound = analyzer.lastHands,
                        bodyFound = analyzer.lastBody,
                        gesture = gesture,
                        frame = analyzer.lastFrame,
                        log = newLog
                    )
                }
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

    fun setSensitivity(v: Int) {
        _ui.value = _ui.value.copy(sensitivity = v)
        if (::analyzer.isInitialized) analyzer.sensitivity = v
    }

    fun snapshotConsumed() {
        _ui.value = _ui.value.copy(snapshotRequest = 0L)
    }

    override fun onCleared() {
        super.onCleared()
        if (::helper.isInitialized) helper.close()
    }
}
