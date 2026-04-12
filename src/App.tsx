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
  
  // Worker-based detection
  const workerRef = useRef<Worker | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(true);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isWorkerReady, setIsWorkerReady] = useState(false);
  
  // HUD state (Floating panels open by default)
  const [isEntityFeedOpen, setIsEntityFeedOpen] = useState(true);

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
  const [detectionHistory, setDetectionHistory] = useState<string[]>([]);
  const [minObjectSize, setMinObjectSize] = useState(40);
  const [detectionInterval, setDetectionInterval] = useState(1); 

  // Mutable refs for high-frequency loops
  const predictionsRef = useRef<DetectedObject[]>([]);
  const trackedTargetRef = useRef<TrackedTarget | null>(null);
  const processingRef = useRef<boolean>(false);

  // Colors
  const colors = {
    primary: '#00F0FF',
    secondary: '#7000FF',
    tracking: '#00FF41',
    locked: '#FF3333', // Tactical Red
    background: 'rgba(8, 12, 24, 0.75)', // Deeper liquid navy
    glass: 'rgba(255, 255, 255, 0.05)',
    accent: 'rgba(0, 240, 255, 0.2)',
  };

  // Initialize Web Worker
  useEffect(() => {
    const worker = new Worker(new URL('./vision.worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (e) => {
      const { type, predictions, message } = e.data;
      if (type === 'MODEL_READY') {
        setIsModelLoading(false);
        setIsWorkerReady(true);
      } else if (type === 'DETECTIONS') {
        predictionsRef.current = predictions;
        processingRef.current = false;

        // Update detection history
        if (predictions.length > 0) {
          setDetectionHistory(prev => {
            const newClasses = (predictions as DetectedObject[]).map(p => p.class);
            const combined = [...new Set([...prev, ...newClasses])];
            return combined.slice(-15); // Keep last 15 unique entities
          });
        }

        // Efficient tracking update
        if (trackedTargetRef.current) {
          const target = trackedTargetRef.current;
          const candidates = (predictions as DetectedObject[]).filter(p => p.class === target.class);
          
          if (candidates.length > 0) {
            const getCenter = (bbox: number[]) => [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2];
            const targetCenter = getCenter(target.smoothBbox);
            
            let bestMatch = candidates[0];
            let minDistance = Infinity;
            
            candidates.forEach((c) => {
              const center = getCenter(c.bbox);
              const dist = Math.hypot(center[0] - targetCenter[0], center[1] - targetCenter[1]);
              if (dist < minDistance) {
                minDistance = dist;
                bestMatch = c;
              }
            });
            target.bbox = bestMatch.bbox;
          }
        }
      } else if (type === 'ERROR') {
        setModelError(message);
        setIsModelLoading(false);
      }
    };

    worker.postMessage({ type: 'INIT' });

    return () => {
      worker.terminate();
    };
  }, []);

  // Setup Camera
  const startCamera = useCallback(async () => {
    if (!videoRef.current) return;
    try {
      setCameraError(null);
      setDetectionHistory([]); // Refresh history on system run
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

      const stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoRef.current.srcObject = stream;
      setIsCameraActive(true);
    } catch (error: any) {
      console.error('Camera Error:', error);
      setCameraError(error.message || 'Unknown camera error');
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

  // Unified Loop for Stats & Worker Capture
  useEffect(() => {
    let frameCount = 0;
    let lastTime = performance.now();
    let animId: number;

    const tick = async () => {
      // FPS Update (every 1s)
      const now = performance.now();
      frameCount++;
      if (now - lastTime >= 1000) {
        setFps(Math.round((frameCount * 1000) / (now - lastTime)));
        frameCount = 0;
        lastTime = now;
      }

      // Worker Inference submission
      if (isWorkerReady && videoRef.current && videoRef.current.readyState === 4 && !processingRef.current) {
        try {
          // Offscreen bitmap creation is fast and zero-copy for worker transfer
          const imageBitmap = await createImageBitmap(videoRef.current);
          processingRef.current = true;
          workerRef.current?.postMessage({
            type: 'DETECT',
            imageBitmap,
            minSize: minObjectSize,
            confidence: 0.35
          }, [imageBitmap]);
        } catch (e) {
          console.error("Bitmap creation error:", e);
        }
      }

      animId = requestAnimationFrame(tick);
    };

    if (isCameraActive && isWorkerReady) {
      animId = requestAnimationFrame(tick);
    }
    return () => cancelAnimationFrame(animId);
  }, [isCameraActive, isWorkerReady, minObjectSize]);

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
    }, 200);
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

          // Draw general objects
          predictionsRef.current.forEach(obj => {
            if (trackedTargetRef.current && trackedTargetRef.current.class === obj.class) return;
            
            ctx.strokeStyle = 'rgba(0, 240, 255, 0.4)';
            ctx.lineWidth = 2;
            ctx.strokeRect(obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]);
            
            ctx.fillStyle = colors.primary;
            ctx.font = '700 10px "Inter", sans-serif';
            ctx.fillText(`${obj.class.toUpperCase()}`, obj.bbox[0] + 4, obj.bbox[1] - 4);
          });

          // Draw tracked target
          if (trackedTargetRef.current) {
            const target = trackedTargetRef.current;
            const alpha = 0.25; 
            target.smoothBbox = [
              target.smoothBbox[0] + alpha * (target.bbox[0] - target.smoothBbox[0]),
              target.smoothBbox[1] + alpha * (target.bbox[1] - target.smoothBbox[1]),
              target.smoothBbox[2] + alpha * (target.bbox[2] - target.smoothBbox[2]),
              target.smoothBbox[3] + alpha * (target.bbox[3] - target.smoothBbox[3])
            ];

            const [x, y, w, h] = target.smoothBbox;
            ctx.strokeStyle = colors.locked;
            ctx.lineWidth = 4;
            ctx.strokeRect(x, y, w, h);
            
            // Corner accents
            const cl = 12;
            ctx.beginPath();
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#FFFFFF';
            ctx.moveTo(x, y+cl); ctx.lineTo(x, y); ctx.lineTo(x+cl, y);
            ctx.moveTo(x+w-cl, y); ctx.lineTo(x+w, y); ctx.lineTo(x+w, y+cl);
            ctx.moveTo(x, y+h-cl); ctx.lineTo(x, y+h); ctx.lineTo(x+cl, y+h);
            ctx.moveTo(x+w-cl, y+h); ctx.lineTo(x+w, y+h); ctx.lineTo(x+w, y+h-cl);
            ctx.stroke();

            // Tactical Label
            ctx.fillStyle = colors.locked;
            ctx.fillRect(x, y-20, ctx.measureText(`LOCKED: ${target.class.toUpperCase()}`).width + 10, 20);
            ctx.fillStyle = '#FFF';
            ctx.fillText(`LOCKED: ${target.class.toUpperCase()}`, x+5, y-6);
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
    if (!videoRef.current) return;
    const rect = videoRef.current.getBoundingClientRect();
    const videoWidth = videoRef.current.videoWidth || 1;
    const videoHeight = videoRef.current.videoHeight || 1;
    const clickX = ((e.clientX - rect.left) / rect.width) * videoWidth;
    const clickY = ((e.clientY - rect.top) / rect.height) * videoHeight;

    const clickedObj = predictionsRef.current.find(obj => {
      const [x, y, w, h] = obj.bbox;
      return clickX >= x && clickX <= x + w && clickY >= y && clickY <= y + h;
    });

    if (clickedObj) handleObjectSelect(clickedObj);
    else clearTracking();
  };

  return (
    <div className="fixed inset-0 bg-black text-white font-sans overflow-hidden flex flex-col">
      {/* FULL SCREEN VIDEO LAYER */}
      <div 
        className="absolute inset-0 z-0 cursor-crosshair overflow-hidden"
        onClick={handleViewportClick}
      >
        <video
          ref={videoRef}
          autoPlay
          muted
          playsInline
          className="w-full h-full object-cover scale-105"
        />
        <canvas 
          ref={canvasRef}
          className="absolute inset-0 w-full h-full object-cover pointer-events-none z-10"
        />
        
        {/* Dynamic Static HUD Deco */}
        <div className="absolute inset-0 pointer-events-none border-[30px] border-black/10 z-0" />
        <div className="absolute top-1/2 left-10 w-20 h-px bg-white/10 hidden md:block" />
        <div className="absolute top-1/2 right-10 w-20 h-px bg-white/10 hidden md:block" />
      </div>

      {/* TOP HUD BAR */}
      <header className="relative z-30 px-6 py-4 flex justify-between items-start pointer-events-none">
        <div className="pointer-events-auto">
          <div className="flex items-center gap-3">
             <div className="p-2 backdrop-blur-md border border-cyan-500/30 rounded-lg" style={{ backgroundColor: colors.accent }}>
                <Aperture className="w-5 h-5 text-cyan-400" />
             </div>
             <div>
                <h1 className="text-xs font-black tracking-[0.3em] uppercase text-white drop-shadow-lg">Mission Control</h1>
                <div className="flex items-center gap-2 mt-1">
                   <div className={`w-1.5 h-1.5 rounded-full ${isCameraActive ? 'bg-cyan-400 animate-pulse' : 'bg-red-500'}`} />
                   <span className="text-[9px] font-bold text-cyan-400/70 uppercase tracking-widest">Vision_Link: {isCameraActive ? 'Active' : 'Standby'}</span>
                </div>
             </div>
          </div>
        </div>
        
        <div className="flex flex-col items-end gap-2 pointer-events-auto">
           <div className="backdrop-blur-md border border-white/10 px-3 py-1.5 rounded-md flex items-center gap-4" style={{ backgroundColor: colors.background }}>
              <span className="text-[10px] font-mono text-white/40 uppercase tracking-widest">Hz_Pulse</span>
              <span className="text-sm font-mono text-cyan-400">{fps}</span>
           </div>
        </div>
      </header>

      {/* FLOATING SIDEBARS - Desktop Only or Collapsed on Mobile */}
      <div className="absolute inset-x-6 top-24 bottom-24 pointer-events-none z-20 hidden md:flex justify-between items-stretch">
      </div>

      {/* RIGHT PANEL: ENTITY FEED - Bottom Right */}
      <AnimatePresence>
        {isEntityFeedOpen && (
          <motion.aside
            initial={{ x: 300, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 300, opacity: 0 }}
            className="absolute bottom-28 right-6 md:right-10 z-30 w-64 backdrop-blur-xl border border-white/10 rounded-2xl p-5 pointer-events-auto flex flex-col gap-4 max-h-[40vh]"
            style={{ backgroundColor: colors.background }}
          >
            <div className="flex justify-between items-center border-b border-white/5 pb-3">
              <h3 className="text-[10px] font-black uppercase tracking-[0.2em] text-white/60 flex items-center gap-2">
                 <Scan className="w-3 h-3" /> Entity_Feed
              </h3>
              <button onClick={() => setIsEntityFeedOpen(false)} className="text-white/20 hover:text-white transition-colors">
                 <X className="w-4 h-4" />
              </button>
            </div>

            <div className="flex-1 overflow-y-auto custom-scrollbar space-y-4">
              {/* Active Detections */}
              <div className="space-y-2">
                <h4 className="text-[8px] font-bold text-cyan-400/50 uppercase tracking-widest">Active_Detections</h4>
                {detectedUIObjects.length === 0 ? (
                  <div className="text-[9px] text-white/10 italic text-center py-4 uppercase tracking-widest">Scanning...</div>
                ) : (
                  detectedUIObjects.map((obj, i) => (
                    <button 
                      key={i} 
                      onClick={() => handleObjectSelect(obj)}
                      className={`w-full flex justify-between items-center p-2 rounded-lg border transition-all ${trackingLabel === obj.class ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-400' : 'bg-white/5 border-white/5 text-white/40 hover:bg-white/10'}`}
                    >
                      <span className="text-[9px] font-bold uppercase tracking-wider">{obj.class}</span>
                      <span className="text-[8px] font-mono opacity-60">{(obj.score*100).toFixed(0)}%</span>
                    </button>
                  ))
                )}
              </div>

              {/* Detection History */}
              <div className="space-y-2 pt-2 border-t border-white/5">
                <h4 className="text-[8px] font-bold text-indigo-400/50 uppercase tracking-widest">Session_History</h4>
                <div className="flex flex-wrap gap-1.5">
                  {detectionHistory.length === 0 ? (
                    <div className="text-[8px] text-white/10 italic uppercase tracking-widest">No history</div>
                  ) : (
                    detectionHistory.map((h, i) => (
                      <span key={i} className="px-2 py-1 bg-white/5 border border-white/5 rounded-md text-[8px] font-bold text-white/40 uppercase tracking-tighter">
                        {h}
                      </span>
                    ))
                  )}
                </div>
              </div>
            </div>
          </motion.aside>
        )}
      </AnimatePresence>

      {/* CONDITIONAL SPATIAL COORDINATES - Top Right */}
      <AnimatePresence>
        {selectedCoords && (
          <motion.div
            initial={{ x: 50, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 50, opacity: 0 }}
            className="absolute top-24 right-6 md:right-10 z-30 p-3 backdrop-blur-xl border border-cyan-500/30 rounded-xl pointer-events-none w-48"
            style={{ backgroundColor: colors.background }}
          >
             <div className="flex flex-col gap-3">
                <div className="flex flex-col gap-0.5 border-b border-white/10 pb-2">
                   <span className="text-[8px] font-black text-cyan-400 uppercase tracking-widest flex items-center gap-2">
                      <Target className="w-2.5 h-2.5" /> Lock_Acquired
                   </span>
                   <span className="text-sm font-black text-white uppercase truncate">{trackingLabel}</span>
                </div>
                <div className="grid grid-cols-2 gap-2">
                   {['X', 'Y', 'W', 'H'].map((d) => (
                      <div key={d} className="flex flex-col bg-white/5 p-1.5 rounded-md">
                         <span className="text-[7px] text-white/30 font-bold">{d}</span>
                         <span className="text-[10px] font-mono text-cyan-50/80">
                            {d === 'X' ? selectedCoords.x : d === 'Y' ? selectedCoords.y : d === 'W' ? selectedCoords.w : selectedCoords.h}
                         </span>
                      </div>
                   ))}
                </div>
             </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* BOTTOM ACTION BAR - Unified Mobile & Desktop */}
      <footer className="relative z-40 p-4 md:p-6 mt-auto">
         <div className="max-w-2xl mx-auto flex items-center justify-center gap-3 md:gap-4">
            <button 
              onClick={() => setShowSettings(true)}
              className="flex-1 md:flex-none p-3 md:px-6 rounded-xl backdrop-blur-lg border border-white/10 hover:bg-white/10 transition-all flex items-center justify-center gap-3"
              style={{ backgroundColor: colors.background }}
            >
               <Settings className="w-4 h-4 text-white/60" />
               <span className="text-[10px] font-black uppercase tracking-[0.2em] hidden md:block">Sys_Conf</span>
            </button>
            <button 
              onClick={() => setCameraFacing(prev => prev === 'user' ? 'environment' : 'user')}
              className="p-4 md:p-5 rounded-full bg-cyan-500 text-black shadow-[0_0_20px_rgba(0,240,255,0.4)] hover:scale-105 active:scale-95 transition-all"
            >
               <RefreshCw className="w-5 h-5" />
            </button>
            <button 
              onClick={() => setShowInfo(!showInfo)}
              className="flex-1 md:flex-none p-3 md:px-6 rounded-xl backdrop-blur-lg border border-white/10 hover:bg-white/10 transition-all flex items-center justify-center gap-3"
              style={{ backgroundColor: colors.background }}
            >
               <Info className="w-4 h-4 text-white/60" />
               <span className="text-[10px] font-black uppercase tracking-[0.2em] hidden md:block">Manual</span>
            </button>
            
            {/* Toggle buttons for sidebars (visible on mobile only) */}
            <div className="md:hidden flex gap-2">
               <button onClick={() => setIsEntityFeedOpen(!isEntityFeedOpen)} className={`p-3 rounded-xl border transition-all ${isEntityFeedOpen ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-400' : 'bg-black/40 border-white/10 text-white/40'}`}>
                  <Scan className="w-4 h-4" />
               </button>
            </div>
         </div>
      </footer>

      {/* Loading & Error States stay same (optimized) */}
      <AnimatePresence>
        {isModelLoading && (
          <motion.div initial={{opacity:0}} animate={{opacity:1}} exit={{opacity:0}} className="fixed inset-0 z-50 bg-black/90 backdrop-blur-2xl flex flex-col items-center justify-center gap-10">
             <div className="relative">
                <motion.div animate={{rotate:360}} transition={{duration:4, repeat:Infinity, ease:"linear"}} className="w-32 h-32 rounded-full border border-dashed border-cyan-500/30" />
                <Aperture className="absolute inset-0 m-auto w-10 h-10 text-cyan-400 animate-pulse" />
             </div>
             <p className="text-xs font-black uppercase tracking-[0.5em] text-cyan-400">Loading Neural Link</p>
          </motion.div>
        )}
      </AnimatePresence>


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

                {/* Min Object Size */}
                <div className="space-y-3">
                  <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-100/50">Min Object Size (px)</label>
                  <div className="flex items-center gap-4">
                    <input 
                      type="range" 
                      min="10" 
                      max="200" 
                      value={minObjectSize} 
                      onChange={(e) => setMinObjectSize(parseInt(e.target.value))}
                      className="flex-1 accent-cyan-400"
                    />
                    <span className="text-xs font-mono text-cyan-400 w-8">{minObjectSize}</span>
                  </div>
                </div>

                {/* Detection Interval */}
                <div className="space-y-3">
                  <label className="text-[10px] font-bold uppercase tracking-[0.2em] text-cyan-100/50">Inference Skip (Frames)</label>
                  <div className="grid grid-cols-4 gap-2">
                    {[1, 2, 3, 4].map((val) => (
                      <button
                        key={val}
                        onClick={() => setDetectionInterval(val)}
                        className={`py-2.5 rounded-xl border text-[10px] font-bold transition-all flex items-center justify-center gap-2 ${
                          detectionInterval === val 
                            ? 'bg-cyan-500/20 border-cyan-500/40 text-cyan-300 shadow-[0_0_15px_rgba(0,240,255,0.1)]' 
                            : 'bg-white/5 border-white/10 text-white/50 hover:bg-white/10'
                        }`}
                      >
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
       
