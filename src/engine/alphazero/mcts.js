// Single-game PUCT MCTS over a Position, using the ONNX net for leaf
// evaluation. Sequential (one inference at a time) — the net is tiny, so a few
// hundred sims fit comfortably in onnxruntime-web.
//
// Sign convention (matches training/quoridor_core/src/mcts.rs):
//   - Node.q is stored from the node's side-to-move POV.
//   - During PUCT selection from a parent, we negate child.q (a high q for
//     the child is a bad outcome for the parent's side).
//   - During backprop, we flip the value sign at each ply walking upward.

import { evaluate } from './net.js'

const C_PUCT = 1.5

class Node {
  constructor(prior = 0.0) {
    this.prior = prior
    this.visitCount = 0
    this.valueSum = 0.0
    this.children = [] // array of { move (policy index), node }
    this.expanded = false
  }
  get q() {
    return this.visitCount > 0 ? this.valueSum / this.visitCount : 0.0
  }
}

function selectChild(node) {
  const sqrtParent = Math.sqrt(Math.max(node.visitCount, 1))
  let best = null
  let bestScore = -Infinity
  for (const c of node.children) {
    const score = -c.node.q + (C_PUCT * c.node.prior * sqrtParent) / (1 + c.node.visitCount)
    if (score > bestScore) {
      bestScore = score
      best = c
    }
  }
  return best
}

// Softmax over the legal moves' logits; returns the net's value.
async function expand(node, pos) {
  const { policy: logits, value } = await evaluate(pos.encode())
  const moves = pos.legalMoves()
  let maxLogit = -Infinity
  for (const m of moves) if (logits[m] > maxLogit) maxLogit = logits[m]
  const exp = moves.map((m) => Math.exp(logits[m] - maxLogit))
  const sum = exp.reduce((a, b) => a + b, 0)
  node.children = moves.map((m, i) => ({ move: m, node: new Node(sum > 0 ? exp[i] / sum : 0) }))
  node.expanded = true
  return value
}

function backprop(path, value) {
  let v = value
  for (let i = path.length - 1; i >= 0; i--) {
    path[i].visitCount += 1
    path[i].valueSum += v
    v = -v
  }
}

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now())

/** Search from `root` (a Position) and return the chosen policy index. */
export async function chooseMove(rootPos, { sims = 400, timeBudgetMs = 1500 } = {}) {
  const root = new Node()
  await expand(root, rootPos)
  if (root.children.length === 1) return root.children[0].move

  const startedAt = now()
  for (let s = 0; s < sims; s++) {
    if (now() - startedAt > timeBudgetMs) break

    const pos = rootPos.clone()
    const path = [root]
    let node = root
    while (node.expanded && node.children.length > 0) {
      const choice = selectChild(node)
      pos.push(choice.move)
      path.push(choice.node)
      node = choice.node
    }

    const tval = pos.terminalValue()
    backprop(path, tval !== null ? tval : await expand(node, pos))
  }

  // Pick the most-visited child. Ties broken by higher q (from our POV).
  let best = null
  let bestVisits = -1
  let bestQ = -Infinity
  for (const c of root.children) {
    const v = c.node.visitCount
    const q = -c.node.q
    if (v > bestVisits || (v === bestVisits && q > bestQ)) {
      bestVisits = v
      bestQ = q
      best = c
    }
  }
  return best.move
}
