// The AlphaZero engine, drop-in replacement for ../random.js.
//
//   await bestMove(state)  // -> a move object (see game.js)
//
// Async because the first call may need to fetch the ONNX model and warm up
// the inference session. Caller must only invoke when it's the bot's turn and
// the game isn't over.

import { chooseMove } from './mcts.js'
import { loadModel } from './net.js'
import { Position } from './position.js'

const SIMS = 400
const TIME_BUDGET_MS = 1500

export async function bestMove(state) {
  const pos = Position.fromGame(state)
  return pos.decode(await chooseMove(pos, { sims: SIMS, timeBudgetMs: TIME_BUDGET_MS }))
}

// Optional: pre-warm the model so the first move doesn't pay the load cost.
export function preload() {
  return loadModel()
}
