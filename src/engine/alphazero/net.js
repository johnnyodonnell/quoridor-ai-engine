// ONNX inference session wrapper. Lazily loads the model on first call and
// caches the session for the lifetime of the page.
//
// Contract:
//   await loadModel(url?)        // optional explicit pre-warm
//   await evaluate(positionF32)  // -> { policy: Float32Array(140), value: number }
//
// vite.config.js serves onnxruntime-web's wasm/jsep assets under /onnx-wasm/.
// We try WebGPU first and fall back to WASM if unavailable.

import * as ort from 'onnxruntime-web'

import { INPUT_CHANNELS, POLICY_SIZE } from './position.js'

ort.env.wasm.wasmPaths = '/onnx-wasm/'

const DEFAULT_MODEL_URL = '/models/current.onnx'

let sessionPromise = null

export function loadModel(url = DEFAULT_MODEL_URL) {
  if (sessionPromise) return sessionPromise
  sessionPromise = (async () => {
    const providers = []
    if (typeof navigator !== 'undefined' && navigator.gpu) providers.push('webgpu')
    providers.push('wasm')
    try {
      return await ort.InferenceSession.create(url, { executionProviders: providers })
    } catch (err) {
      console.warn('AlphaZero: WebGPU failed, falling back to WASM', err)
      return await ort.InferenceSession.create(url, { executionProviders: ['wasm'] })
    }
  })()
  return sessionPromise
}

export async function evaluate(positionFloat32) {
  const session = await loadModel()
  const tensor = new ort.Tensor('float32', positionFloat32, [1, INPUT_CHANNELS, 9, 9])
  const out = await session.run({ position: tensor })
  const policy = out.policy.data
  const value = out.value.data[0]
  if (policy.length !== POLICY_SIZE) {
    throw new Error(`AlphaZero: expected policy length ${POLICY_SIZE}, got ${policy.length}`)
  }
  return { policy, value }
}
