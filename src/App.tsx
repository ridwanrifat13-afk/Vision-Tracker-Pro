/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useEffect, useRef, useState, useCallback } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';
import { Camera, Target, Info, RefreshCw, MapPin, Scan, Aperture, Activity, Settings, Check, X } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// --- Types ---
interface DetectedObject {
  bbox: [number, number, number, number]; // [x, y, width, height]
  class: string;
  score: number;
}

interface TrackedTarget {
  class: string;
  bbox: [number, number, number, number];
  smoothBbox: [number, number, number, number];
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const [model, setModel] = useState<cocoSsd.ObjectDetection | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isCameraActive, setIsCameraActive] = useState(false);
  
  // React state for UI (updated throttled)
  const [detectedUIObjects, setDetectedUIObjects] = useState<DetectedObject[]>([]);
  const [trackingLabel, setTrackingLabel] = useState<string | null>(null);
  const [fps, setFps] = useState(0);
  
  const [showInfo, setShowInfo] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [cameraFacing, setCameraFacing] = useState<'user' | 'environment'>('environment');
  const [targetFps, setTargetFps] = useState<number>(60);
  const [targetResolution, setTargetResolution] = useState<'480p' | '720p' | '1080p'>('720p');
  const [modelError, setModelError] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [selectedCoords, setSelectedCoords] = useState<{x:number, y:number, w:number, h:number} | null>(null);

  // Mutable refs for high-frequency loops (no React re-renders)
  const predictionsRef = useRef<DetectedObject[]>([]);
  const trackedTargetRef = useRef<TrackedTarget | null>(null);

  // Colors
  const colors = {
    primary: '#00F0FF', // Cyan
    secondary: '#7000FF', // Deep Indigo
    tracking: '#00FF41', // Accent Green for hit
    background: 'rgba(10, 15, 30, 0.8)',
  };

  // Load COCO-SSD Model with Retry Logic
  const loadModel = useCallback(async (retryCount = 0) => {
    try {
      setIsModelLoading(true);
      setModelError(null);
      await tf.ready();
      
      const loadedModel = await cocoSsd.load({ base: 'mobilenet_v2' });
      setModel(loadedModel);
      setIsModelLoading(false);
    } catch (error) {
      console.error(`Model load attempt ${retryCount + 1} failed:`, error);
      if (retryCount < 2) {
        setTimeout(() => loadModel(retryCount + 1), 2000 * (retryCount + 1));
      } else {
        setModelError('Neural weights fetch failed. Please check network/ad-blockers.');
        setIsModelLoading(false);
      }
    }
  }, []);

  useEffect(() => { loadModel(); }, [loadModel]);

  // Setup Camera
  const startCamera = useCallback(async () => {
    if (!videoRef.current) return;
    try {
      setCameraError(null);
      
      const resMap = {
        '480p': { width: 640, height: 480 },
        '720p': { width: 1280, height: 720 },
        '1080p': { width: 1920, height: 1080 }
      };
      
      const constraints = {
        video: { 
          facingMode: cameraFacing, 
          width: { ideal: resMap[targetResolution].width }, 
          height: { ideal: resMap[targetResolution].height },
          frameRate: { ideal: targetFps }
        },
        audio: false,
      };

      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('BROWSER_UNSUPPORTED');
      }

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoRef.current.srcObject = stream;
      setIsCameraActive(true);
    } catch (error: any) {
      console.error('Error accessing camera:', error);
      
      if (error.name === 'NotAllowedError' || error.name === 'PermissionDeniedError' || error.message?.toLowerCase().includes('denied')) {
        setCameraError('Camera access was denied. To fix this:\n1. Click the lock icon 🔒 in your browser address bar.\n2. Change "Camera" to "Allow".\n3. Refresh the page or click "Retry Connection".\n\nNote: If you are in a private/incognito window, you may need to enable permissions manually.');
      } else if (error.name === 'NotFoundError' || error.name === 'DevicesNotFoundError') {
        setCameraError('No camera found. Please ensure your camera is connected and not being used by another app.');
      } else if (error.message === 'BROWSER_UNSUPPORTED') {
        setCameraError('Your browser does not support camera access or you are in an insecure context. Please try a different browser or open the app in a new tab.');
      } else {
        setCameraError(`Camera Error: ${error.message || 'Unknown error'}. Please ensure no other application is using the camera.`);
      }
      
      setIsCameraActive(false);
    }
  }, [cameraFacing, targetResolution, targetFps]);

  useEffect(() => {
    startCamera();
    return () => {
      if (videoRef.current?.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [startCamera]);

  // Detection Loop (Runs as fast as model allows)
  useEffect(() => {
    let active = true;
    let frameCount = 0;
    let lastTime = performance.now();

    const detect = async () => {
      if (!active) return;
      
      if (model && videoRef.current && videoRef.current.readyState === 4) {
        try {
          // Broadened object identification: 50 max objects, 35% minimum confidence
          const predictions = await model.detect(videoRef.current, 50, 0.35);
          predictionsRef.current = predictions;

          if (trackedTargetRef.current) {
            const currentTrackedClass = trackedTargetRef.current.class;
            const candidates = predictions.filter((p: any) => p.class === currentTrackedClass);
            
            if (candidates.length > 0) {
              const getCenter = (bbox: number[]) => [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2];
              const targetCenter = getCenter(trackedTargetRef.current.smoothBbox);
              
              let bestMatch = candidates[0];
              let minDistance = Infinity;
              
              candidates.forEach((c: any) => {
                const center = getCenter(c.bbox);
                const dist = Math.hypot(center[0] - targetCenter[0], center[1] - targetCenter[1]);
                if (dist < minDistance) {
                  minDistance = dist;
                  bestMatch = c;
                }
              });

              trackedTargetRef.current.bbox = bestMatch.bbox;
            }
          }

          // FPS Calc
          frameCount++;
          const now = performance.now();
          if (now - lastTime >= 1000) {
            setFps(Math.round((frameCount * 1000) / (now - lastTime)));
            frameCount = 0;
            lastTime = now;
          }
        } catch (err) {
          console.error("Detection error:", err);
        }
      }
      // Use setTimeout to respect target FPS
      const delay = 1000 / targetFps;
      setTimeout(detect, delay); 
    };

    if (isCameraActive && !isModelLoading && model) {
      detect();
    }
    return () => { active = false; };
  }, [model, isCameraActive, isModelLoading, targetFps]);

  // UI Throttle Loop (Updates React UI state sparingly to save performance)
  useEffect(() => {
    const interval = setInterval(() => {
      setDetectedUIObjects([...predictionsRef.current]);
      
      if (trackedTargetRef.current) {
        setSelectedCoords({
          x: Math.round(trackedTargetRef.current.smoothBbox[0]),
          y: Math.round(trackedTargetRef.current.smoothBbox[1]),
          w: Math.round(trackedTargetRef.current.smoothBbox[2]),
          h: Math.round(trackedTargetRef.current.smoothBbox[3])
        });
      } else {
        setSelectedCoords(null);
      }
    }, 250);
    return () => clearInterval(interval);
  }, []);

  // Visual Render Loop (Hardware Accelerated Canvas drawing at 60fps)
  useEffect(() => {
    let animationFrameId: number;

    const render = () => {
      if (canvasRef.current && videoRef.current) {
        const ctx = canvasRef.current.getContext('2d');
        if (ctx) {
          const vWidth = videoRef.current.videoWidth || 1;
          const vHeight = videoRef.current.videoHeight || 1;
          
          if (canvasRef.current.width !== vWidth || canvasRef.current.height !== vHeight) {
            canvasRef.current.width = vWidth;
            canvasRef.current.height = vHeight;
          }

          ctx.clearRect(0, 0, vWidth, vHeight);

          // Draw general unselected objects
          predictionsRef.current.forEach(obj => {
            if (trackedTargetRef.current && trackedTargetRef.current.class === obj.class) {
              return; // Handled separately below
            }
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
            ctx.lineWidth = 2;
            ctx.strokeRect(obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]);
            
            // Draw Label background
            ctx.fillStyle = 'rgba(0, 0, 0, 0.5)';
            ctx.fillRect(obj.bbox[0] - 1, obj.bbox[1] - 22, ctx.measureText(`${obj.class.toUpperCase()} ${(obj.score*100).toFixed(0)}%`).width + 12, 22);
            // Draw Label Text
            ctx.fillStyle = colors.primary;
            ctx.font = 'bold 11px "Courier New", Courier, monospace';
            ctx.fillText(`${obj.class.toUpperCase()} ${(obj.score*100).toFixed(0)}%`, obj.bbox[0] + 5, obj.bbox[1] - 7);
          });

          // Process Tracked Target with Exponential Moving Average (EMA) smoothing
          if (trackedTargetRef.current) {
            const target = trackedTargetRef.current;
            const alpha = 0.25; // Smoothness factor (lower = smoother but more delay)

            target.smoothBbox = [
              target.smoothBbox[0] + alpha * (target.bbox[0] - target.smoothBbox[0]),
              target.smoothBbox[1] + alpha * (target.bbox[1] - target.smoothBbox[1]),
              target.smoothBbox[2] + alpha * (target.bbox[2] - target.smoothBbox[2]),
              target.smoothBbox[3] + alpha * (target.bbox[3] - target.smoothBbox[3])
            ];

            const [x, y, w, h] = target.smoothBbox;

            // Target Highlight Box Core
            ctx.strokeStyle = colors.primary;
            ctx.lineWidth = 3;
            ctx.strokeRect(x, y, w, h);
            
            // Add aesthetic corner accents
            const cornerLength = 15;
            ctx.beginPath();
            ctx.lineWidth = 4;
            ctx.strokeStyle = '#FFFFFF';
            
            // Top Left
            ctx.moveTo(x, y + cornerLength); ctx.lineTo(x, y); ctx.lineTo(x + cornerLength, y);
            // Top Right
            ctx.moveTo(x + w - cornerLength, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + cornerLength);
            // Bottom Left
            ctx.moveTo(x, y + h - cornerLength); ctx.lineTo(x, y + h); ctx.lineTo(x + cornerLength, y + h);
            // Bottom Right
            ctx.moveTo(x + w - cornerLength, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - cornerLength);
            ctx.stroke();

            // Label
            const labelText = `LOCKED TRACKING: ${target.class.toUpperCase()}`;
            ctx.font = 'bold 12px "Courier New", Courier, monospace';
            const metrics = ctx.measureText(labelText);
            ctx.fillStyle = colors.primary;
            ctx.fillRect(x - 2, y - 28, metrics.width + 16, 26);
            ctx.fillStyle = '#000000';
            ctx.fillText(labelText, x + 6, y - 10);

            // Crosshair overlay inside block
            ctx.beginPath();
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.6)';
            ctx.lineWidth = 1.5;
            const cx = x + w / 2;
            const cy = y + h / 2;
            ctx.moveTo(cx - 12, cy); ctx.lineTo(cx + 12, cy);
            ctx.moveTo(cx, cy - 12); ctx.lineTo(cx, cy + 12);
            ctx.stroke();
          }
        }
      }
      animationFrameId = requestAnimationFrame(render);
    };

    render();
    return () => cancelAnimationFrame(animationFrameId);
  }, []);

  const handleObjectSelect = (obj: DetectedObject) => {
    setTrackingLabel(obj.class);
    trackedTargetRef.current = { 
      class: obj.class, 
      bbox: [...obj.bbox] as [number, number, number, number],
      smoothBbox: [...obj.bbox] as [number, number, number, number]
    };
  };

  const clearTracking = () => {
    setTrackingLabel(null);
    trackedTargetRef.current = null;
  }

  const handleViewportClick = (e: React.MouseEvent) => {
    if (!videoRef.current || isModelLoading) return;
    
    // Check if the click is on the actual underlying feed size
    const rect = videoRef.current.getBoundingClientRect();
    const videoWidth = videoRef.current.videoWidth || 1;
    const videoHeight = videoRef.current.videoHeight || 1;
    
    const clickX = ((e.clientX - rect.left) / rect.width) * videoWidth;
    const clickY = ((e.clientY - rect.top) / rect.height) * videoHeight;

    const sortedObjects = [...predictionsRef.current].sort((a, b) => (a.bbox[2] * a.bbox[3]) - (b.bbox[2] * b.bbox[3]));
    
    const clickedObj = sortedObjects.find(obj => {
      const [x, y, w, h] = obj.bbox;
      return clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h;
    });

    if (clickedObj) handleObjectSelect(clickedObj);
    else clearTracking();
  };

  return (
    <div className="min-h-screen bg-[#05050A] text-white font-sans overflow-hidden flex flex-col selection:bg-cyan-500/30">
      {/* Dynamic Ambient Background */}
      <div className="fixed inset-0 z-0 pointer-events-none opacity-40">
        <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] rounded-full bg-cyan-900/40 blur-[120px]" />
        <div className="absolute bottom-[-20%] right-[-10%] w-[50%] h-[50%] rounded-full bg-indigo-900/30 blur-[120px]" />
      </div>

      {/* Header / HUD panel */}
      <header className="px-6 py-4 flex justify-between items-center bg-white/5 backdrop-blur-xl border-b border-white/10 z-20 shadow-[0_4px_30px_rgba(0,0,0,0.3)]">
        <div className="flex items-center gap-4">
          <div className="relative w-10 h-10 rounded-xl bg-gradient-to-br from-cyan-400/20 to-indigo-500/20 flex items-center justify-center border border-cyan-400/30 shadow-[0_0_15px_rgba(0,240,255,0.2)]">
            <Aperture className="w-5 h-5 text-cyan-400" />
          </div>
          <div>
            <h1 className="text-base font-black tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400 drop-shadow-sm uppercase">VisionTracker Pro</h1>
            <div className="flex items-center gap-3 text-[11px] font-medium text-cyan-200/50 uppercase tracking-widest mt-0.5">
              <span className="flex items-center gap-1.5">
                <div className={`w-1.5 h-1.5 rounded-full ${isCameraActive ? 'bg-cyan-400 shadow-[0_0_8px_rgba(0,240,255,0.8)] animate-pulse' : 'bg-rose-500'}`} />
                {isCameraActive ? 'Live Visual Sync' : 'Offline Mode'}
              </span>
              <span className="w-1 h-1 rounded-full bg-white/10" />
              <span className="flex items-center gap-1">
                <Activity className="w-3 h-3 text-indigo-400" /> FPS: {fps.toString().padStart(2, '0')}
              </span>
            </div>
          </div>
        </div>
        <div className="flex gap-3">
          <button 
            onClick={() => setShowSettings(true)}
            className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all hover:scale-105 active:scale-95 text-cyan-50/70 hover:text-cyan-400"
            title="Settings"
          >
            <Settings className="w-4 h-4" />
          </button>
          <button 
            onClick={() => setCameraFacing(prev => prev === 'user' ? 'environment' : 'user')}
            className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all hover:scale-105 active:scale-95 text-cyan-50/70 hover:text-cyan-400"
            title="Flip Camera"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
          <button 
            onClick={() => setShowInfo(!showInfo)}
            className="p-2.5 rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 transition-all hover:scale-105 active:scale-95 text-cyan-50/70 hover:text-cyan-400"
            title="System Info"
          >
            <Info className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Video Viewport Element */}
      <main 
        className="relative flex-1 overflow-hidden flex items-center justify-center cursor-crosshair z-10 m-4 rounded-2xl border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.4)] bg-black/50 backdrop-blur-sm group"
        onClick={handleViewportClick}
      >
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="absolute inset-0 w-full h-full object-cover opacity-80 mix-blend-screen"
        />
        
        {/* Hardware Accelerated Canvas Overlay */}
        <canvas 
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover pointer-events-none"
        />

        {/* Camera Error / Permission Requirement Request */}
        {!isCameraActive && !isModelLoading && (
          <div className="absolute inset-0 z-40 bg-black/60 backdrop-blur-lg flex flex-col items-center justify-center p-6 text-center">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-white/5 to-white/10 flex items-center justify-center border border-white/10 mb-6 glass-panel">
              <Camera className="w-10 h-10 text-white/50" />
            </div>
            <h3 className="text-lg font-black uppercase tracking-widest mb-3 bg-clip-text text-transparent bg-gradient-to-r from-red-400 to-rose-300">Camera Access Required</h3>
            <div className="text-sm text-white/60 max-w-md mb-8 leading-relaxed font-light whitespace-pre-line">
              {cameraError || "The application requires camera access to process spatial data in real-time."}
            </div>
            <button 
              onClick={startCamera}
              className="px-8 py-3.5 bg-cyan-500/10 hover:bg-cyan-500/20 text-cyan-400 border border-cyan-500/30 text-xs font-bold uppercase tracking-[0.2em] rounded-xl transition-all hover:shadow-[0_0_20px_rgba(0,240,255,0.2)] flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" /> {cameraError ? "Retry Connection" : "Enable Bridge"}
            </button>
          </div>
        )}

        {/* Clear Tracking Button Floating within Feed Feed */}
        <AnimatePresence>
          {trackingLabel && (
            <motion.button
              initial={{ opacity: 0, scale: 0.9, y: -20 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.9, y: -20 }}
              onClick={(e) => { e.stopPropagation(); clearTracking(); }}
              className="absolute top-6 left-1/2 -translate-x-1/2 z-30 px-6 py-3 bg-red-500/10 hover:bg-red-500/20 backdrop-blur-xl border border-red-500/30 text-red-300 rounded-full text-[10px] font-bold uppercase tracking-widest flex items-center gap-2 shadow-xl transition-all hover:scale-105 active:scale-95"
            >
              <RefreshCw className="w-4 h-4" /> Terminate Tracking Lock
            </motion.button>
          )}
        </AnimatePresence>

        {/* Loading State Element */}
        <AnimatePresence>
          {isModelLoading && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="absolute inset-0 z-50 bg-[#05050A]/90 backdrop-blur-2xl flex flex-col items-center justify-center gap-8"
            >
              <div className="relative">
                <div className="w-32 h-32 rounded-full flex items-center justify-center relative">
                  <div className="absolute inset-0 bg-cyan-400/5 rounded-full blur-[20px] animate-pulse" />
                  <motion.div 
                    animate={{ rotate: 360 }}
                    transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
                    className="absolute inset-0 rounded-full border border-dashed border-cyan-500/40"
                  />
                  <motion.div 
                    animate={{ rotate: -360 }}
                    transition={{ duration: 4, repeat: Infinity, ease: "linear" }}
                    className="absolute inset-2 rounded-full border border-transparent border-t-indigo-500/60 border-b-cyan-500/60 opacity-60"
                  />
                  <Scan className="w-10 h-10 text-cyan-400" />
                </div>
              </div>
              <div className="text-center space-y-3">
                <p className="text-base tracking-[0.4em] uppercase font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400">Initializing Neural Engine</p>
                <div className="flex items-center justify-center gap-2">
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                  <div className="w-2 h-2 bg-cyan-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                  <div className="w-2 h-2 bg-indigo-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
                </div>
                <p className="text-xs text-white/40 uppercase tracking-widest font-mono mt-4">Loading Core Vision Weights</p>
              </div>
            </motion.div>
          )}

          {modelError && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute inset-0 z-50 bg-[#05050A]/95 backdrop-blur-2xl flex flex-col items-center justify-center p-8 text-center"
            >
              <div className="w-20 h-20 rounded-2xl bg-red-500/10 flex items-center justify-center border border-red-500/20 mb-8 shadow-[0_0_30px_rgba(239,68,68,0.15)]">
                <Target className="w-10 h-10 text-red-500" />
              </div>
              <h2 className="text-xl font-black uppercase tracking-[0.2em] mb-4 text-red-400 drop-shadow-md">Critical System Error</h2>
              <p className="text-sm text-white/60 max-w-sm mb-10 leading-relaxed font-light">
                {modelError}
              </p>
              <button 
                onClick={() => loadModel()}
                className="px-8 py-3.5 bg-white/5 hover:bg-white/10 text-white border border-white/20 rounded-xl text-xs font-bold uppercase tracking-[0.2em] transition-all flex items-center gap-2"
              >
                <RefreshCw className="w-4 h-4" /> Force Reboot Server
              </button>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* Footer Data Panel (Glassmorphic Setup) */}
      <footer className="h-44 px-4 pb-4 pt-2 z-20 shrink-0">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 h-full">
          {/* Spatial Coordinates Info Panel */}
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-3 md:p-4 flex flex-row md:flex-col gap-3 md:gap-2 relative overflow-hidden group">
            <div className="absolute top-0 right-0 w-32 h-32 bg-cyan-500/5 rounded-full blur-[40px] -mr-10 -mt-10 transition-opacity group-hover:opacity-100 opacity-50" />
            <div className="w-[35%] md:w-full flex flex-col justify-center border-r md:border-r-0 md:border-b border-white/10 pr-3 md:pr-0 md:pb-2 relative z-10 shrink-0">
              <span className="text-[8px] md:text-[10px] font-bold uppercase tracking-[0.1em] md:tracking-[0.2em] text-cyan-100/50 flex items-center gap-1 md:gap-1.5">
                <MapPin className="w-2.5 h-2.5 md:w-3 md:h-3 text-cyan-400" /> <span className="truncate">Spatial Data</span>
              </span>
              {selectedCoords && (
                <span className="text-[8px] md:text-[10px] font-bold text-cyan-400 tracking-tighter md:tracking-widest bg-cyan-400/10 px-1.5 md:px-2 py-0.5 rounded-md border border-cyan-400/20 mt-1 inline-block w-fit truncate">
                  {trackingLabel?.toUpperCase()}
                </span>
              )}
            </div>
            <div className="flex-1 grid grid-cols-2 lg:grid-cols-4 gap-1.5 md:gap-3 relative z-10 overflow-y-auto md:overflow-visible custom-scrollbar">
              {['X', 'Y', 'W', 'H'].map((dim) => {
                const val = selectedCoords ? (dim === 'X' ? selectedCoords.x : dim === 'Y' ? selectedCoords.y : dim === 'W' ? selectedCoords.w : selectedCoords.h) : null;
                return (
                  <div key={dim} className="flex flex-col justify-center bg-black/20 rounded-lg md:rounded-xl p-1.5 md:p-3 border border-white/5">
                    <span className="text-[7px] md:text-[9px] font-bold text-white/30 uppercase tracking-widest mb-0.5 md:mb-1.5">{dim}</span>
                    <span className={`text-xs md:text-xl font-mono tracking-tight font-light ${val !== null ? 'text-cyan-50 drop-shadow-[0_0_8px_rgba(0,240,255,0.4)]' : 'text-white/20'}`}>
                      {val !== null ? val.toString().padStart(4, '0') : '0000'}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Environmental Detections Feed List */}
          <div className="bg-white/5 backdrop-blur-xl rounded-2xl border border-white/10 p-3 md:p-4 flex flex-row md:flex-col relative overflow-hidden">
            <div className="absolute bottom-0 left-0 w-40 h-40 bg-indigo-500/5 rounded-full blur-[40px] -ml-10 -mb-10" />
            <div className="w-[35%] md:w-full flex flex-col justify-center border-r md:border-r-0 md:border-b border-white/10 pr-3 md:pr-0 md:pb-2 md:mb-3 relative z-10 shrink-0">
              <span className="text-[8px] md:text-[10px] font-bold uppercase tracking-[0.1em] md:tracking-[0.2em] text-cyan-100/50 flex items-center gap-1 md:gap-1.5">
                <Scan className="w-2.5 h-2.5 md:w-3 md:h-3 text-indigo-400" /> <span className="truncate">Entities</span>
              </span>
              <span className="text-[8px] md:text-[10px] font-bold text-indigo-300 bg-indigo-500/10 px-1.5 md:px-2.5 py-0.5 rounded-md border border-indigo-500/20 mt-1 inline-block w-fit">
                {detectedUIObjects.length}
              </span>
            </div>
            <div className="flex-1 overflow-y-auto pr-1 md:pr-2 custom-scrollbar relative z-10 md:-mx-1 md:px-1 md:mt-1 ml-2 md:ml-0">
              {detectedUIObjects.length === 0 ? (
                <div className="h-full flex items-center justify-center text-[10px] md:text-xs text-white/30 font-light italic tracking-wide">
                  Scanning...
                </div>
              ) : (
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-1.5 md:gap-2.5 pb-1">
                  {detectedUIObjects.map((obj, i) => {
                    const isSelected = trackingLabel === obj.class;
                    return (
                      <button
                        key={`${obj.class}-${i}`}
                        onClick={() => handleObjectSelect(obj)}
                        className={`w-full text-left px-2 py-1.5 md:px-3 md:py-2.5 rounded-lg md:rounded-xl text-[8px] md:text-[10px] flex justify-between items-center transition-all ${
                          isSelected 
                            ? 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-[0_0_15px_rgba(0,240,255,0.15)] scale-[1.02]' 
                            : 'bg-black/40 text-white/70 border border-white/5 hover:bg-white/10 hover:border-white/20 hover:text-white'
                        }`}
                      >
                        <span className="uppercase font-bold tracking-wider truncate mr-1">{obj.class}</span>
                        <span className={`font-mono ${isSelected ? 'opacity-100' : 'opacity-40'}`}>{(obj.score * 100).toFixed(0)}%</span>
                      </button>
                    )
                  })}
                </div>
              )}
            </div>
          </div>
        </div>
      </footer>

      {/* Manual Info Overlay Component */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-[60] flex items-center justify-center p-6"
          >
            <div className="absolute inset-0 bg-[#05050A]/80 backdrop-blur-xl" onClick={() => setShowSettings(false)} />
            <motion.div 
              initial={{ scale: 0.95, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 20 }}
              className="relative bg-black/40 border border-white/10 p-8 rounded-3xl max-w-md w-full shadow-[0_20px_50px_rgba(0,0,0,0.5)] overflow-hidden backdrop-blur-2xl"
            >
              <div className="absolute top-0 right-0 w-64 h-64 bg-indigo-500/10 rounded-full blur-[60px] pointer-events-none" />
              <div className="flex justify-between items-center mb-6">
                <h2 className="text-xl font-black flex items-center gap-3 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400 uppercase tracking-widest">
                  <Settings className="w-6 h-6 text-cyan-400" /> System Config
                </h2>
                <button onClick={() => setShowSettings(false)} className="text-white/40 hover:text-white">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-8">
                {/* FPS Settings */}
                <div className="space-y-3">
                  <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-100/50">Target Frame Rate (FPS)</label>
                  <div className="grid grid-cols-3 gap-2">
                    {[20, 30, 60].map((val) => (
                      <button
                        key={val}
                        onClick={() => setTargetFps(val)}
                        className={`py-2.5 rounded-xl border text-[10px] font-bold transition-all flex items-center justify-center gap-2 ${
                          targetFps === val 
                            ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300 shadow-[0_0_15px_rgba(0,240,255,0.1)]' 
                            : 'bg-white/5 border-white/10 text-white/50 hover:bg-white/10'
                        }`}
                      >
                        {targetFps === val && <Check className="w-3 h-3" />}
                        {val} FPS
                      </button>
                    ))}
                  </div>
                </div>

                {/* Resolution Settings */}
                <div className="space-y-3">
                  <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-100/50">Visual Resolution</label>
                  <div className="grid grid-cols-3 gap-2">
                    {(['480p', '720p', '1080p'] as const).map((val) => (
                      <button
                        key={val}
                        onClick={() => setTargetResolution(val)}
                        className={`py-2.5 rounded-xl border text-[10px] font-bold transition-all flex items-center justify-center gap-2 ${
                          targetResolution === val 
                            ? 'bg-indigo-500/20 border-indigo-500/40 text-indigo-300 shadow-[0_0_15px_rgba(112,0,255,0.1)]' 
                            : 'bg-white/5 border-white/10 text-white/50 hover:bg-white/10'
                        }`}
                      >
                        {targetResolution === val && <Check className="w-3 h-3" />}
                        {val}
                      </button>
                    ))}
                  </div>
                </div>

                <div className="bg-white/5 p-4 rounded-2xl border border-white/5">
                  <p className="text-[10px] text-white/40 leading-relaxed italic">
                    Note: Higher resolutions and frame rates increase neural processing load and battery consumption. 
                    Changes to resolution will restart the camera bridge.
                  </p>
                </div>
              </div>

              <button 
                onClick={() => setShowSettings(false)}
                className="mt-8 w-full py-4 bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 hover:from-cyan-500/30 hover:to-indigo-500/30 backdrop-blur-md border border-cyan-500/30 text-cyan-300 font-bold uppercase text-xs tracking-[0.2em] rounded-xl transition-all shadow-lg active:scale-[0.98]"
              >
                Apply Configuration
              </button>
            </motion.div>
          </motion.div>
        )}

        {showInfo && (
           <motion.div
           initial={{ opacity: 0 }}
           animate={{ opacity: 1 }}
           exit={{ opacity: 0 }}
           className="fixed inset-0 z-[60] flex items-center justify-center p-6"
         >
           <div className="absolute inset-0 bg-[#05050A]/80 backdrop-blur-xl" onClick={() => setShowInfo(false)} />
           <motion.div 
             initial={{ scale: 0.95, y: 20 }}
             animate={{ scale: 1, y: 0 }}
             exit={{ scale: 0.95, y: 20 }}
             className="relative bg-black/40 border border-white/10 p-8 rounded-3xl max-w-md w-full shadow-[0_20px_50px_rgba(0,0,0,0.5)] overflow-hidden backdrop-blur-2xl"
           >
             <div className="absolute top-0 right-0 w-64 h-64 bg-cyan-500/10 rounded-full blur-[60px] pointer-events-none" />
             <h2 className="text-xl font-black mb-6 flex items-center gap-3 text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-indigo-400 uppercase tracking-widest">
               <Info className="w-6 h-6 text-cyan-400" /> System Manual
             </h2>
             <div className="space-y-6 text-sm text-white/60 leading-relaxed font-light">
               <div className="bg-white/5 p-5 rounded-2xl border border-white/5 shadow-inner">
                 <strong className="text-cyan-50 block mb-2 uppercase tracking-[0.1em] text-[11px]">Neural Broadened Engine</strong>
                 <p>The system utilizes an amplified neural network parsing through dozens of concurrent items dynamically, dramatically widening identification targets in low-visibility or highly active environments.</p>
               </div>
               <div className="bg-white/5 p-5 rounded-2xl border border-white/5 shadow-inner">
                 <strong className="text-cyan-50 block mb-2 uppercase tracking-[0.1em] text-[11px]">Hardware Canvas Tracking</strong>
                 <p>Tap any detected item on the real-time feed or within the entity index to initialize a lock. Engine leverages raw hardware-accelerated EMA algorithms yielding 60fps tracking smoothness independent of detection rate lag.</p>
               </div>
             </div>
             <button 
               onClick={() => setShowInfo(false)}
               className="mt-8 w-full py-4 bg-gradient-to-r from-cyan-500/20 to-indigo-500/20 hover:from-cyan-500/30 hover:to-indigo-500/30 backdrop-blur-md border border-cyan-500/30 text-cyan-300 font-bold uppercase text-xs tracking-[0.2em] rounded-xl transition-all shadow-lg active:scale-[0.98]"
             >
               Initialize System
             </button>
           </motion.div>
         </motion.div>
        )}
      </AnimatePresence>
      <style>{`
        .custom-scrollbar::-webkit-scrollbar { width: 6px; }
        .custom-scrollbar::-webkit-scrollbar-track { background: rgba(0, 0, 0, 0.2); border-radius: 8px;}
        .custom-scrollbar::-webkit-scrollbar-thumb { background: rgba(255, 255, 255, 0.1); border-radius: 8px; border: 1px solid rgba(0,0,0,0.5);}
        .custom-scrollbar::-webkit-scrollbar-thumb:hover { background: rgba(0, 240, 255, 0.3); }
      `}</style>
    </div>
  );
}
