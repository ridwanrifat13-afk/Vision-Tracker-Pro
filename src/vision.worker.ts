import * as tf from '@tensorflow/tfjs';
import * as cocoSsd from '@tensorflow-models/coco-ssd';

let model: cocoSsd.ObjectDetection | null = null;

// Initialize TensorFlow.js and the model
async function init() {
  try {
    await tf.ready();
    // Using mobilenet_v2 for better balance of speed and accuracy
    model = await cocoSsd.load({ base: 'mobilenet_v2' });
    
    // Warm up the model with a dummy tensor
    const dummy = tf.zeros([300, 300, 3], 'int32');
    await model.detect(dummy as any);
    tf.dispose(dummy);
    
    (self as any).postMessage({ type: 'MODEL_READY' });
  } catch (error) {
    console.error('Worker Model Load Error:', error);
    (self as any).postMessage({ type: 'ERROR', message: 'Failed to load vision engine.' });
  }
}

self.onmessage = async (e: MessageEvent) => {
  const { type, imageBitmap, minSize, confidence } = e.data;

  if (type === 'INIT') {
    await init();
  } else if (type === 'DETECT' && model && imageBitmap) {
    try {
      // Direct detection on ImageBitmap is supported by coco-ssd
      const predictions = await model.detect(imageBitmap, 50, confidence || 0.25);
      
      // Filter predictions based on size on the worker thread
      const filtered = predictions.filter(
        p => (p.bbox[2] * p.bbox[3]) > (minSize * minSize)
      );

      // Return predictions and inform the main thread we're ready for the next frame
      (self as any).postMessage({ 
        type: 'DETECTIONS', 
        predictions: filtered 
      }, [imageBitmap]); // Transfer back if needed or let it be closed by the browser
      
      imageBitmap.close(); // Clean up the ImageBitmap
    } catch (err) {
      console.error('Detection Error in Worker:', err);
      imageBitmap.close();
    }
  }
};
