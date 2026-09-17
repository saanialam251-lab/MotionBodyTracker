package com.motiontracker

import android.content.ContentValues
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.slideInVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.platform.LocalLifecycleOwner
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.viewmodel.compose.viewModel
import com.motiontracker.ui.OverlayView
import com.motiontracker.vision.HandGesture
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent { TrackerApp() }
    }
}

private val SeaDark = Color(0xFF0B1220)
private val SeaMid = Color(0xFF16233D)
private val SkeletonGreen = Color(0xFF00E676)
private val MotionRed = Color(0xFFF85149)
private val MotionGreen = Color(0xFF3FB950)

@Composable
fun TrackerApp(vm: TrackerViewModel = viewModel()) {
    val ui by vm.ui.collectAsState()
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val analyzerExecutor = remember { Executors.newSingleThreadExecutor() }

    var previewView by remember { mutableStateOf<PreviewView?>(null) }
    var overlay by remember { mutableStateOf<OverlayView?>(null) }
    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, android.Manifest.permission.CAMERA) ==
                PackageManager.PERMISSION_GRANTED
        )
    }
    val permLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    val notifPermLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { /* no state to track — beep just won't fire if denied */ }

    LaunchedEffect(Unit) {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            val granted = ContextCompat.checkSelfPermission(
                context, android.Manifest.permission.POST_NOTIFICATIONS
            ) == PackageManager.PERMISSION_GRANTED
            if (!granted) notifPermLauncher.launch(android.Manifest.permission.POST_NOTIFICATIONS)
        }
    }

    var showEnrollDialog by remember { mutableStateOf(false) }
    var enrollName by remember { mutableStateOf("") }

    LaunchedEffect(ui.frame) {
        ui.frame?.let {
            overlay?.frame = it
            overlay?.invalidate()
        }
    }

    LaunchedEffect(ui.snapshotRequest) {
        if (ui.snapshotRequest == 0L) return@LaunchedEffect
        val pv = previewView
        if (pv != null) {
            try {
                val src = pv.bitmap
                if (src != null) {
                    val base = src.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
                    overlay?.draw(android.graphics.Canvas(base))
                    saveSnapshot(context, base)
                    Toast.makeText(context, "Snapshot saved", Toast.LENGTH_SHORT).show()
                }
            } catch (_: Exception) {
                Toast.makeText(context, "Snapshot failed", Toast.LENGTH_SHORT).show()
            }
        }
        vm.snapshotConsumed()
    }

    LaunchedEffect(ui.frontCamera, hasPermission, ui.modelsReady) {
        if (!hasPermission || !ui.modelsReady) return@LaunchedEffect
        try {
            val provider = ProcessCameraProvider.getInstance(context).get()
            provider.unbindAll()
            val preview = Preview.Builder().build().also { p ->
                previewView?.let { pv -> p.setSurfaceProvider(pv.surfaceProvider) }
            }
            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_YUV_420_888)
                .build()
                .also { it.setAnalyzer(analyzerExecutor, vm.analyzer) }
            val selector = if (ui.frontCamera) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }
            provider.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
        } catch (e: Exception) {
            Toast.makeText(context, "Camera error: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    val bgShift by rememberInfiniteTransition(label = "bg").animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            tween(6000, easing = LinearEasing),
            RepeatMode.Reverse
        ),
        label = "bgShift"
    )
    val bg = Brush.verticalGradient(
        listOf(SeaMid.copy(alpha = 0.6f + bgShift * 0.3f), SeaDark, SeaDark)
    )

    Box(Modifier.fillMaxSize().background(bg)) {
        if (hasPermission) {
            AndroidView(
                factory = { ctx ->
                    BoxLayout(ctx).also { box ->
                        previewView = box.first
                        overlay = box.second
                    }
                },
                modifier = Modifier.fillMaxSize()
            )
        } else {
            Column(
                Modifier.align(Alignment.Center),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Camera permission needed", color = Color.White)
                Spacer(Modifier.height(12.dp))
                Button(onClick = { permLauncher.launch(android.Manifest.permission.CAMERA) }) {
                    Text("Grant permission")
                }
            }
        }

        Row(
            Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
                .background(Brush.verticalGradient(listOf(Color.Black.copy(alpha = 0.65f), Color.Transparent)))
                .horizontalScroll(rememberScrollState())
                .padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            TrackerButton(if (ui.running) "Stop" else "Start") {
                if (!hasPermission) {
                    permLauncher.launch(android.Manifest.permission.CAMERA)
                } else {
                    vm.setRunning(!ui.running)
                }
            }
            TrackerButton("Flip", enabled = ui.running) { vm.flipCamera() }
            ToggleButton("Hands", ui.handsOn) { vm.toggleHands() }
            ToggleButton("Body", ui.bodyOn) { vm.toggleBody() }
            ToggleButton("Motion", ui.motionOn) { vm.toggleMotion() }
            ToggleButton("Gestures", ui.gesturesOn) { vm.toggleGestures() }
            ToggleButton("Face", ui.faceOn && ui.faceModelReady) { vm.toggleFace() }
            TrackerButton("Enroll", enabled = ui.faceOn && ui.facesFound > 0) {
                enrollName = ""
                showEnrollDialog = true
            }
            TrackerButton("Shot", enabled = ui.running) {
                val pv = previewView ?: return@TrackerButton
                try {
                    val src = pv.bitmap ?: return@TrackerButton
                    val base = src.copy(android.graphics.Bitmap.Config.ARGB_8888, true)
                    overlay?.draw(android.graphics.Canvas(base))
                    saveSnapshot(context, base)
                    Toast.makeText(context, "Snapshot saved", Toast.LENGTH_SHORT).show()
                } catch (_: Exception) {
                    Toast.makeText(context, "Snapshot failed", Toast.LENGTH_SHORT).show()
                }
            }
        }

        LaunchedEffect(ui.enrollMessage) {
            ui.enrollMessage?.let {
                Toast.makeText(context, it, Toast.LENGTH_SHORT).show()
                vm.clearEnrollMessage()
            }
        }

        if (showEnrollDialog) {
            AlertDialog(
                onDismissRequest = { showEnrollDialog = false },
                title = { Text("Enroll this face") },
                text = {
                    OutlinedTextField(
                        value = enrollName,
                        onValueChange = { enrollName = it },
                        label = { Text("Name") },
                        singleLine = true
                    )
                },
                confirmButton = {
                    Button(onClick = {
                        vm.enrollFace(enrollName)
                        showEnrollDialog = false
                    }) { Text("Save") }
                },
                dismissButton = {
                    Button(onClick = { showEnrollDialog = false }) { Text("Cancel") }
                }
            )
        }

        Column(
            Modifier
                .align(Alignment.TopStart)
                .padding(start = 12.dp, top = 76.dp)
                .background(Color.Black.copy(alpha = 0.82f), RoundedCornerShape(12.dp))
                .padding(12.dp)
        ) {
            Text(
                "Hands: ${ui.handsFound}  ·  Body: ${if (ui.bodyFound) "yes" else "no"}",
                color = Color(0xFF9DB4D8),
                fontSize = MaterialTheme.typography.bodySmall.fontSize
            )
            Spacer(Modifier.height(4.dp))
            Text(
                "Gesture: ${if (ui.gesture == HandGesture.NONE) "—" else ui.gesture.name.replace('_', ' ')}",
                color = Color(0xFF9DB4D8),
                fontSize = MaterialTheme.typography.bodySmall.fontSize
            )
            if (ui.faceModelReady) {
                Spacer(Modifier.height(4.dp))
                val faceText = if (ui.facesFound == 0) {
                    "Faces: 0"
                } else {
                    "Faces: ${ui.facesFound} (${ui.faceNames.joinToString(", ")})"
                }
                Text(faceText, color = Color(0xFF9DB4D8), fontSize = MaterialTheme.typography.bodySmall.fontSize)
            }
            Spacer(Modifier.height(6.dp))
            Text("Motion: ${ui.motionPercent}%", color = Color.White)
            val fill by animateFloatAsState(
                targetValue = (ui.motionPercent * 4).coerceAtMost(100) / 100f,
                label = "fill"
            )
            Spacer(Modifier.height(4.dp))
            Box(
                Modifier
                    .width(150.dp)
                    .height(8.dp)
                    .background(SeaMid, RoundedCornerShape(4.dp))
            ) {
                Box(
                    Modifier
                        .fillMaxWidth(fill)
                        .fillMaxHeight()
                        .background(
                            if (ui.motionHot) MotionRed else MotionGreen,
                            RoundedCornerShape(4.dp)
                        )
                )
            }
            Spacer(Modifier.height(8.dp))
            Text(
                "Sensitivity",
                color = Color(0xFF9DB4D8),
                fontSize = MaterialTheme.typography.bodySmall.fontSize
            )
            Slider(
                value = ui.sensitivity.toFloat(),
                onValueChange = { vm.setSensitivity(it.toInt()) },
                valueRange = 1f..100f,
                modifier = Modifier.width(150.dp)
            )
        }

        val alertAlpha by rememberInfiniteTransition(label = "alert").animateFloat(
            initialValue = 0.35f,
            targetValue = 1f,
            animationSpec = infiniteRepeatable(
                tween(500, easing = FastOutSlowInEasing),
                RepeatMode.Reverse
            ),
            label = "alertAlpha"
        )
        AnimatedVisibility(
            visible = ui.motionHot,
            enter = scaleIn(tween(300)) + fadeIn(),
            exit = scaleOut(tween(250)) + fadeOut(),
            modifier = Modifier.align(Alignment.Center)
        ) {
            Text(
                "MOTION DETECTED",
                color = MotionRed.copy(alpha = alertAlpha),
                style = MaterialTheme.typography.headlineSmall
            )
        }

        Column(
            Modifier
                .align(Alignment.BottomStart)
                .padding(12.dp)
                .background(Color.Black.copy(alpha = 0.85f), RoundedCornerShape(12.dp))
                .padding(horizontal = 12.dp, vertical = 8.dp)
                .widthIn(max = 300.dp)
        ) {
            Text(
                "Palm = start · Peace = shot · Fist = stop",
                color = Color(0xFF7C93B2),
                fontSize = MaterialTheme.typography.bodySmall.fontSize
            )
            ui.log.forEach { entry ->
                var visible by remember(entry) { mutableStateOf(false) }
                LaunchedEffect(entry) { visible = true }
                AnimatedVisibility(
                    visible = visible,
                    enter = slideInVertically { it } + fadeIn()
                ) {
                    Text(
                        entry,
                        color = Color(0xFF9DB4D8),
                        fontSize = MaterialTheme.typography.bodySmall.fontSize,
                        modifier = Modifier.padding(vertical = 2.dp)
                    )
                }
            }
        }

        if (!ui.modelsReady) {
            Text(
                "Model files missing — put hand_landmarker.task and pose_landmarker.task in app/src/main/assets (see README)",
                color = MotionRed,
                modifier = Modifier.align(Alignment.Center).padding(24.dp)
            )
        } else if (!ui.faceModelReady) {
            Text(
                "Face detection off — add face_detection_short_range.tflite to app/src/main/assets to enable it (see README)",
                color = Color(0xFF7C93B2),
                fontSize = MaterialTheme.typography.bodySmall.fontSize,
                modifier = Modifier.align(Alignment.TopCenter).padding(top = 132.dp, start = 24.dp, end = 24.dp)
            )
        }
    }

    DisposableEffect(Unit) {
        onDispose { analyzerExecutor.shutdown() }
    }
}

private class BoxLayout(context: android.content.Context) : android.widget.FrameLayout(context) {
    val first: PreviewView = PreviewView(context).apply {
        scaleType = PreviewView.ScaleType.FILL_CENTER
        implementationMode = PreviewView.ImplementationMode.COMPATIBLE
    }
    val second: OverlayView = OverlayView(context)
    init {
        addView(first, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        addView(second, LayoutParams(LayoutParams.MATCH_PARENT, LayoutParams.MATCH_PARENT))
        second.setWillNotDraw(false)
    }
}

@Composable
private fun TrackerButton(text: String, enabled: Boolean = true, onClick: () -> Unit) {
    val scale by animateFloatAsState(if (enabled) 1f else 0.95f, label = "s")
    Button(
        onClick = onClick,
        enabled = enabled,
        modifier = Modifier.scale(scale),
        colors = ButtonDefaults.buttonColors(containerColor = SeaMid, contentColor = Color.White),
        shape = RoundedCornerShape(10.dp),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp)
    ) { Text(text, fontSize = MaterialTheme.typography.bodySmall.fontSize) }
}

@Composable
private fun ToggleButton(text: String, active: Boolean, onClick: () -> Unit) {
    Button(
        onClick = onClick,
        colors = ButtonDefaults.buttonColors(
            containerColor = if (active) SkeletonGreen.copy(alpha = 0.85f) else SeaMid,
            contentColor = if (active) SeaDark else Color.White
        ),
        shape = RoundedCornerShape(10.dp),
        contentPadding = PaddingValues(horizontal = 12.dp, vertical = 8.dp)
    ) { Text(text, fontSize = MaterialTheme.typography.bodySmall.fontSize) }
}

private fun saveSnapshot(context: android.content.Context, bmp: android.graphics.Bitmap) {
    val values = ContentValues().apply {
        put(MediaStore.Images.Media.DISPLAY_NAME, "tracker_${System.currentTimeMillis()}.png")
        put(MediaStore.Images.Media.MIME_TYPE, "image/png")
        if (Build.VERSION.SDK_INT >= 29) {
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/MotionTracker")
        }
    }
    val uri = context.contentResolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return
    context.contentResolver.openOutputStream(uri)?.use {
        bmp.compress(android.graphics.Bitmap.CompressFormat.PNG, 100, it)
    }
}
